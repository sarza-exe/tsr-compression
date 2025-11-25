import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import time

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Architecture Imports ---
from Architectures.cnn_6x2 import SimpleCNN_6x2
from Architectures.enhanced_lenet5 import EnhancedLeNet5
from Architectures.resnet50_custom import ResNet50Custom
from Architectures.efficientnet_b0_custom import EfficientNetB0Custom

# --- Helper Functions Imports ---
from data_original import val_loader
from Training.train_utils import validate
from Compression.metrics_utils import measure_latency

# --- Configuration ---
QUANTIZED_MODELS_DIR = "../Compressed_Models/Quantization"
DEVICE = "cpu"  # Quantized models in PyTorch MUST run on CPU
NUM_CLASSES = 43

# --- Model Registry ---
MODEL_CLASSES = {
    "SimpleCNN_6x2": SimpleCNN_6x2,
    "EnhancedLeNet5": EnhancedLeNet5,
    "ResNet50Custom": ResNet50Custom,
    "EfficientNetB0Custom": EfficientNetB0Custom
}


# ==========================================
# Wrapper Class (Must match training!)
# ==========================================
class QuantizedModelWrapper(nn.Module):
    def __init__(self, model_to_wrap):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model_to_wrap
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ==========================================
# Model Reconstruction Logic
# ==========================================
def load_quantized_model(model_name, method, filepath):
    """
    Recreates the quantized model structure so state_dict can be loaded.
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")

    # 1. Initialize Float Model
    ModelClass = MODEL_CLASSES[model_name]
    model = ModelClass(num_classes=NUM_CLASSES)

    # 2. Prepare structure based on method
    if method == "Dynamic":
        # Dynamic only affects Linear layers
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    elif method in ["Static", "QAT"]:
        # Static/QAT requires Wrapping -> Fusing -> Configuring -> Preparing -> Converting
        model = QuantizedModelWrapper(model)
        model.eval()

        # --- REPLICATE FUSING LOGIC (Must match quantization.py exactly) ---
        try:
            if model_name == "SimpleCNN_6x2":
                torch.ao.quantization.fuse_modules(model.model, [
                    ['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3'],
                    ['conv4', 'bn4'], ['conv5', 'bn5'], ['conv6', 'bn6']
                ], inplace=True)
            elif model_name == "EnhancedLeNet5":
                torch.ao.quantization.fuse_modules(model.model, [
                    ['conv1', 'bn1'], ['conv2', 'bn2']
                ], inplace=True)
        except Exception:
            pass

            # Configuration & Preparation (Using 'x86' or 'qnnpack' as generic fallback)
        model.qconfig = torch.quantization.get_default_qconfig('x86')
        torch.quantization.prepare(model, inplace=True)

        # Conversion (Creates the Int8 structure)
        torch.quantization.convert(model, inplace=True)

    # 3. Load Weights
    # Load to CPU
    checkpoint = torch.load(filepath, map_location="cpu")

    # Handle different saving formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"      Strict loading failed, trying non-strict. Error: {e}")
        model.load_state_dict(state_dict, strict=False)

    return model


# ==========================================
# Report Generation
# ==========================================
def generate_markdown_report(results):
    """Prints a formatted markdown table."""

    # Sort by validation accuracy (highest first)
    results.sort(key=lambda x: x["val_acc"], reverse=True)

    print("\n\n---")
    print("## Quantization Benchmark Report")
    print("\nModels benchmarked on CPU (INT8 inference).")
    print("\n| Model | Method | Val. Acc. | Size (MB) | Latency (ms) |")
    print("|:---|:---|:---|:---|:---|")

    for r in results:
        print(f"| {r['model']} | {r['method']} | **{r['val_acc']:.2f}%** | {r['size_mb']:.2f} | {r['latency']:.3f} |")
    print("\n---")


# ==========================================
# Main Execution
# ==========================================
def main():
    if not os.path.exists(QUANTIZED_MODELS_DIR):
        print(f"Error: Directory {QUANTIZED_MODELS_DIR} does not exist.")
        return

    print(f"--- Starting Quantization Benchmark (using {DEVICE}) ---")

    files = [f for f in os.listdir(QUANTIZED_MODELS_DIR) if f.endswith(".pt")]
    results = []

    print(f"Found {len(files)} models in {QUANTIZED_MODELS_DIR}.\n")

    for filename in files:
        filepath = os.path.join(QUANTIZED_MODELS_DIR, filename)

        try:
            # Parse filename: quantized_ModelName_Method.pt
            # Expected format from your quantization script
            parts = filename.replace(".pt", "").split("_")

            # parts[0] is "quantized"
            method = parts[-1]  # Last part is method (Dynamic, Static, QAT)
            model_name = "_".join(parts[1:-1])  # Middle part is model name (e.g., SimpleCNN_6x2)

            if model_name not in MODEL_CLASSES:
                print(f"Skipping {filename}: Unknown model class '{model_name}'")
                continue

            print(f"Benchmarking: {model_name} | {method}")

            # 1. Load Model
            model = load_quantized_model(model_name, method, filepath)
            model.to(DEVICE)  # Always CPU for quantization

            # 2. Measure Size (Disk size is the real metric for quantization)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)

            # 3. Measure Latency
            # Note: measure_latency inside metrics_utils might try to use get_input_size logic.
            # Ensure metrics_utils handles the Wrapper name or use a simple local measurement here.
            latency = measure_latency(model, device=DEVICE) * 1000  # Convert to ms

            # 4. Measure Accuracy
            criterion = nn.CrossEntropyLoss()
            _, val_acc = validate(model, val_loader, criterion, torch.device(DEVICE))

            # Append results
            results.append({
                "model": model_name,
                "method": method,
                "val_acc": val_acc * 100,  # Convert to percentage
                "size_mb": size_mb,
                "latency": latency
            })

            print(f"   -> Acc: {val_acc:.4f} | Size: {size_mb:.2f}MB | Latency: {latency:.2f}ms")

        except Exception as e:
            print(f"   Error benchmarking {filename}: {e}")

    if results:
        generate_markdown_report(results)
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()