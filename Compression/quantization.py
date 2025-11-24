import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Helper Functions Imports ---
from data_original import train_loader, val_loader
from Training.train_utils import train_one_epoch, validate
from metrics_utils import count_params, count_flops, measure_latency

# --- Configuration ---
# We do NOT set torch.backends.quantized.engine globally to avoid Windows compatibility issues.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "../Compressed_Models/Quantization"
os.makedirs(SAVE_DIR, exist_ok=True)

# QAT Settings
QAT_EPOCHS = 2
LR = 1e-4
NUM_CLASSES = 43


# --- Helper: Get File Size ---
def get_model_size_mb(model_path):
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0.0


# ==========================================
# Wrapper Class for Quantization
# ==========================================
class QuantizedModelWrapper(nn.Module):
    """
    Wraps the model to add QuantStub and DeQuantStub.
    This is CRITICAL for Static Quantization (PTQ) and QAT.
    It converts input Floats -> INT8 before the first layer,
    and INT8 -> Floats after the last layer.
    """

    def __init__(self, model_to_wrap):
        super().__init__()
        self.quant = torch.quantization.QuantStub()  # Converts Float to QInt8
        self.model = model_to_wrap  # Original Architecture
        self.dequant = torch.quantization.DeQuantStub()  # Converts QInt8 back to Float

    def forward(self, x):
        # Manually define the flow of data
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ==========================================
# 1. Dynamic Quantization
# ==========================================
def apply_dynamic_quantization(model):
    """
    Converts weights of Linear layers to INT8.
    Activations are quantized dynamically during inference.
    No wrapper needed here as Conv layers remain Float32.
    """
    print("   -> Applying Dynamic Quantization...")
    model.cpu()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


# ==========================================
# 2. Static Post-Training Quantization (PTQ)
# ==========================================
def apply_static_quantization(model, data_loader):
    """
    Converts both weights and activations to INT8.
    Uses calibration data to determine ranges.
    """
    print("   -> Applying Static Quantization...")

    # 1. Wrap the model to handle Float input -> Quantized Conv errors
    model = QuantizedModelWrapper(model)

    model.cpu()
    model.eval()

    # Get the inner model class name
    model_name = model.model.__class__.__name__

    # 2. Fusing Modules (Optimization)
    # We act on 'model.model' because 'model' is the wrapper.
    try:
        if model_name == "SimpleCNN_6x2":
            # Fusing Conv+BN. ReLU is functional in this architecture, so it's not fused here.
            torch.ao.quantization.fuse_modules(model.model, [
                ['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3'],
                ['conv4', 'bn4'], ['conv5', 'bn5'], ['conv6', 'bn6']
            ], inplace=True)

        elif model_name == "EnhancedLeNet5":
            torch.ao.quantization.fuse_modules(model.model, [
                ['conv1', 'bn1'], ['conv2', 'bn2']
            ], inplace=True)

        print("      -> Fusing completed.")
    except Exception as e:
        print(f"      ⚠️ Warning: Fusing failed. Error: {e}")

    # 3. Configuration
    # 'x86' is a safe default for PC CPUs.
    model.qconfig = torch.quantization.get_default_qconfig('x86')

    # 4. Prepare
    torch.quantization.prepare(model, inplace=True)

    # 5. Calibrate
    print("      -> Calibrating...")
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= 100: break
            model(images)

    # 6. Convert
    torch.quantization.convert(model, inplace=True)

    return model


# ==========================================
# 3. Quantization Aware Training (QAT)
# ==========================================
def apply_qat(model, train_loader, device):
    """
    Simulates quantization during training (fine-tuning).
    """
    print("   -> Applying Quantization Aware Training (QAT)...")

    # 1. Wrap the model (Important for consistency)
    model = QuantizedModelWrapper(model)

    model.to(device)
    model.train()

    # 2. Config for QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('x86')

    # 3. Prepare QAT
    torch.quantization.prepare_qat(model, inplace=True)

    # 4. Fine-tuning Loop
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"      -> Fine-tuning for {QAT_EPOCHS} epochs...")
    for epoch in range(1, QAT_EPOCHS + 1):
        # Custom training loop for wrapped model
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"         Epoch {epoch} complete. Loss: {running_loss / len(train_loader):.4f}")

    # 5. Convert to CPU and INT8
    model.cpu()
    model.eval()
    torch.quantization.convert(model, inplace=True)

    return model


# ==========================================
# Main Experiment Loop
# ==========================================
def run_experiment(ModelClass):
    results = []

    model_name = ModelClass.__name__
    print(f"\n\n=== Processing Model: {model_name} ===")

    methods = ["Dynamic", "Static", "QAT"]

    for method in methods:
        print(f"\n--- Method: {method} ---")

        # --- SKIP LOGIC FOR UNSUPPORTED ARCHITECTURES ---
        # ResNet fails due to 'add' (skip connection)
        # EfficientNet fails due to 'silu' activation in QuantizedCPU backend
        if ("ResNet" in model_name or "EfficientNet" in model_name) and method in ["Static", "QAT"]:
            print(
                f"   [SKIP] {method} Quantization is not supported for standard {model_name} on this backend (missing 'add' or 'silu' support).")
            continue

        # 1. Load fresh FP32 model
        try:
            # Initialize clean model
            model = ModelClass(num_classes=NUM_CLASSES)
            pretrained_path = f"../Models/{model_name}_best.pt"

            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"]

            # Clean unnecessary keys from state_dict (from previous experiments)
            keys_to_remove = [k for k in state_dict.keys() if "total_ops" in k or "total_params" in k]
            for k in keys_to_remove: del state_dict[k]

            model.load_state_dict(state_dict, strict=False)
        except FileNotFoundError:
            print(f"Skipping {model_name}, checkpoint not found.")
            continue

        # 2. Apply Strategy
        start_time = time.time()

        if method == "Dynamic":
            q_model = apply_dynamic_quantization(model)
        elif method == "Static":
            q_model = apply_static_quantization(model, train_loader)
        elif method == "QAT":
            q_model = apply_qat(model, train_loader, DEVICE)

        processing_time = time.time() - start_time

        # 3. Save & Measure
        q_model.cpu()
        save_name = f"quantized_{model_name}_{method}.pt"
        save_path = os.path.join(SAVE_DIR, save_name)

        # Save state_dict
        torch.save(q_model.state_dict(), save_path)
        file_size_mb = get_model_size_mb(save_path)

        print("      -> Validating accuracy (on CPU)...")
        criterion = nn.CrossEntropyLoss()
        # Quantized models must run on CPU
        _, val_acc = validate(q_model, val_loader, criterion, torch.device("cpu"))

        print("      -> Measuring latency...")
        latency = measure_latency(q_model, device="cpu")

        print(
            f"   [RESULT] {method}: Acc={val_acc:.4f} | Size={file_size_mb:.2f}MB | Latency={latency * 1000:.2f}ms")

        results.append({
            "model": model_name,
            "method": method,
            "val_acc": val_acc,
            "size_mb": file_size_mb,
            "latency": latency,
            "process_time": processing_time
        })

    return results