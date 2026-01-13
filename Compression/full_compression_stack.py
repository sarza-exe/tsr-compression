import os
import sys
import torch
import torch.nn as nn
import pandas as pd

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports
from Architectures.cnn_6x2 import SimpleCNN_6x2
from Architectures.enhanced_lenet5 import EnhancedLeNet5
from Architectures.resnet50_custom import ResNet50Custom
from Architectures.efficientnet_b0_custom import EfficientNetB0Custom

from data_original import val_loader, train_loader
from Training.train_utils import validate
from Compression.metrics_utils import measure_latency, get_input_size
from Compression.slimming_utils import physically_prune_structured


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIR = "../Compressed_Models/Pruned_slimmed"
OUTPUT_DIR = "../Compressed_Models/Full_stack"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CLASSES = {
    "SimpleCNN_6x2": SimpleCNN_6x2,
    "EnhancedLeNet5": EnhancedLeNet5,
    "ResNet50Custom": ResNet50Custom,
    "EfficientNetB0Custom": EfficientNetB0Custom,
}

# ResNet/Efficient -> Dynamic, rest -> Static
QUANT_STRATEGY = {
    "SimpleCNN_6x2": "Static",
    "EnhancedLeNet5": "Static",
    "ResNet50Custom": "Dynamic",
    "EfficientNetB0Custom": "Dynamic"
}

TEACHER_PATH = "../Models/SimpleCNN_6x2_best.pt"


# 1. Logic to Load "Slimmed" Architectures
def load_slimmed_architecture(model_name, pruning_type, amount, filepath):
    """
    Creates a model instance that matches the shape of the saved weights.
    """
    ModelClass = MODEL_CLASSES[model_name]
    # 1. Create standard full model
    model = ModelClass(num_classes=43)

    # 2. If structured pruning was used, we must RESHAPE the model
    # to match the saved weights dimensions.
    if pruning_type == "structured":
        # We assume the saved model was pruned with 'physically_prune_structured'
        input_size = get_input_size(model)
        # Apply the exact same pruning operation to resize layers
        model = physically_prune_structured(model, pruning_ratio=amount, example_input_size=input_size)

    # 3. Load the weights (Now shapes should match)
    checkpoint = torch.load(filepath, map_location="cpu")

    # Handle both full checkpoint dict and direct state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    return model

def load_teacher(model_name, device):
    print(f"   -> Selecting teacher for student: {model_name}")

    ModelClass = MODEL_CLASSES[model_name]
    model = ModelClass(num_classes=43).to(device)

    teacher_path = f"../Models/{model_name}_best.pt"
    print(f"   -> Loading teacher weights from: {teacher_path}")

    ckpt = torch.load(teacher_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    print(f"   -> Teacher {model_name} loaded and set to eval mode")

    return model


def distill(student, teacher, loader, device, epochs=5, T=6.0, alpha=0.7):
    print("   -> Starting Knowledge Distillation")
    print(f"      Epochs: {epochs} | T: {T} | Alpha: {alpha}")
    print(f"      Device: {device}")

    student.train()
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Student forward
            s_logits = student(x)

            # Teacher forward
            with torch.no_grad():
                t_logits = teacher(x)

            # Distillation loss
            loss = (
                alpha * nn.KLDivLoss(reduction="batchmean")(
                    nn.functional.log_softmax(s_logits / T, dim=1),
                    nn.functional.softmax(t_logits / T, dim=1)
                ) * (T * T)
                + (1 - alpha) * nn.functional.cross_entropy(s_logits, y)
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(
                    f"      [KD] Epoch {epoch+1}/{epochs} "
                    f"Step {i}/{len(loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(loader)
        print(f"   -> KD Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    student.eval()
    print("   -> Knowledge Distillation completed")

# 2. Quantization Logic
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


def apply_quantization(model, method, loader):
    model.eval()

    if method == "Dynamic":
        print("      -> Applying Dynamic Quantization...")
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    elif method == "Static":
        print("      -> Applying Static Quantization...")
        model = QuantizedModelWrapper(model)

        # Fusing
        model_name = model.model.__class__.__name__
        try:
            # Note: Fusing might fail if pruning removed layers, so we wrap in try
            if "SimpleCNN" in model_name:
                torch.ao.quantization.fuse_modules(model.model, [
                    ['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3'],
                    ['conv4', 'bn4'], ['conv5', 'bn5'], ['conv6', 'bn6']
                ], inplace=True)
            elif "LeNet" in model_name:
                torch.ao.quantization.fuse_modules(model.model, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
        except Exception as e:
            print(f"      Fusing skipped/failed (expected for pruned models): {e}")

        model.qconfig = torch.quantization.get_default_qconfig('x86')
        torch.quantization.prepare(model, inplace=True)

        # Calibrate
        print("      -> Calibrating...")
        with torch.no_grad():
            for i, (img, _) in enumerate(loader):
                if i >= 50: break
                model(img)

        torch.quantization.convert(model, inplace=True)

    return model


# Main Loop
def main():
    print(f"--- Processing Slimmed Models from {INPUT_DIR}")
    results = []
    criterion = nn.CrossEntropyLoss()

    # Find files ending with _slimmed.pt
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_slimmed.pt")]

    if not files:
        print(f"No models found in {INPUT_DIR}. Make sure filenames end with '_slimmed.pt'")
        return

    for filename in files:
        try:
            # Parse filename assuming format: "ModelName_method_amount_slimmed.pt"
            # Example: SimpleCNN_6x2_structured_0.5_slimmed.pt

            # Remove suffix
            clean_name = filename.replace("_slimmed.pt", "")
            parts = clean_name.split("_")

            # Extract info (assuming last part is amount, second to last is method)
            amount = float(parts[-1])
            p_type = parts[-2]

            # Reconstruct model name (everything before method)
            model_name = "_".join(parts[:-2])

            if model_name not in MODEL_CLASSES:
                print(f"Skipping unknown model: {model_name} (derived from {filename})")
                continue

            # Determine Quantization Strategy
            q_method = QUANT_STRATEGY.get(model_name, "Dynamic")

            print(f"\nProcessing: {model_name} | Pruning: {p_type} {amount} | Target Quant: {q_method}")

            # 1. Load the Slimmed Model
            model = load_slimmed_architecture(model_name, p_type, amount, os.path.join(INPUT_DIR, filename))
            model.to(DEVICE)

            teacher = load_teacher(model_name, DEVICE)

            distill(model, teacher, train_loader, DEVICE)
            model.cpu()

            # 2. Apply Quantization
            model = apply_quantization(model, q_method, train_loader)

            # 3. Save Final Artifact
            save_name = f"Final_{model_name}_Pruned{p_type[0].upper()}{amount}_KD_Quant{q_method}.pt"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            torch.save(model.state_dict(), save_path)

            # 4. Measure Metrics
            print("      -> Measuring metrics...")
            _, acc = validate(model, val_loader, criterion, "cpu")
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            lat = measure_latency(model, device="cpu") * 1000

            results.append({
                "Model": model_name,
                "Pipeline": f"P({p_type[0]}{amount}) + KD + Q({q_method})",
                "Val. Acc.": f"{acc * 100:.2f}%",
                "Size (MB)": f"{size_mb:.2f}",
                "Latency (ms)": f"{lat:.2f}"
            })
            print(f"   Done. Acc: {acc:.2f} | Size: {size_mb:.2f}MB")

        except Exception as e:
            print(f"   Error processing {filename}: {e}")

    # Final Report
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Val. Acc.", ascending=False)

        print("\n=== Pipeline Results ===")
        print(df.to_markdown(index=False))
        df.to_csv(os.path.join(OUTPUT_DIR, "pipeline_results.csv"), index=False)


if __name__ == "__main__":
    main()