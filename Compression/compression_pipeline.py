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
DEVICE = "cpu"  # Quantization always CPU
INPUT_DIR = "../Compressed_Models/Pruned_slimmed"
OUTPUT_DIR = "../Compressed_Models/Pipeline"
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

            # 2. Apply Quantization
            model = apply_quantization(model, q_method, train_loader)

            # 3. Save Final Artifact
            save_name = f"Final_{model_name}_Pruned{p_type[0].upper()}{amount}_Quant{q_method}.pt"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            torch.save(model.state_dict(), save_path)

            # 4. Measure Metrics
            print("      -> Measuring metrics...")
            _, acc = validate(model, val_loader, criterion, DEVICE)
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            lat = measure_latency(model, device=DEVICE) * 1000

            results.append({
                "Model": model_name,
                "Pipeline": f"P({p_type[0]}{amount}) + Q({q_method})",
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