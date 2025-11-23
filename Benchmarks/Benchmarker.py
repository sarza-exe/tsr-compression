import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Compression')))

# Model architectures
from Architectures.cnn_6x2 import SimpleCNN_6x2
from Architectures.efficientnet_b0_custom import EfficientNetB0Custom
from Architectures.enhanced_lenet5 import EnhancedLeNet5
from Architectures.resnet50_custom import ResNet50Custom


# Importing Helper Functions
from data_original import val_loader
from Training.train_utils import validate
from Compression.metrics_utils import count_params, count_flops, measure_latency, get_input_size
from Compression.slimming_utils import physically_prune_structured, make_pruning_permanent

# Model registry
MODEL_CLASSES = {
    "SimpleCNN_6x2": SimpleCNN_6x2,
    "EfficientNetB0Custom": EfficientNetB0Custom,
    "EnhancedLeNet5" : EnhancedLeNet5,
    "ResNet50Custom": ResNet50Custom,
}

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORIGINAL_MODELS_DIR = "../Models"
PRUNED_MODELS_DIR = "../Compressed_Models/Pruned_normal"
NUM_CLASSES = 43  # For GTSRB


# --- Pruning helper functions ---
def apply_unstructured(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def apply_structured(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    return model


# --- Benchmarking Function ---
def run_benchmark(model, val_loader, criterion, device):
    """Runs all benchmarks on a given model."""
    model.eval()
    with torch.no_grad():
        _, val_acc = validate(model, val_loader, criterion, device)

    # Run metrics
    params = count_params(model)
    flops = count_flops(model)

    # Measure latency on CPU for a stable, comparable benchmark
    # Change to `device=DEVICE` to measure GPU latency
    latency = measure_latency(model, device="cpu")

    return {
        "val_acc": val_acc,
        "params": params,
        "flops": flops,
        "latency": latency
    }


# --- Report Generation ---
def generate_markdown_report(results):
    """Prints a formatted markdown table from the results list."""

    # Sort by validation accuracy (highest first)
    results.sort(key=lambda x: x["val_acc"], reverse=True)

    print("\n\n---")
    print("## Benchmark Report")
    print("\nModels benchmarked against GTSRB validation set.")
    print("\n| Model | Method | Pruning % | Val. Acc. | Params (M) | FLOPs (G) | Latency (ms) |")
    print("|:---|:---|:---|:---|:---|:---|:---|")

    for r in results:
        # Format for readability
        model_name = r["model"]
        method = r["method"]
        amount_pct = f"{r['amount'] * 100:.0f}%" if r["amount"] > 0 else "N/A"
        acc_pct = f"{r['val_acc'] * 100:.2f}%"
        params_m = f"{r['params'] / 1e6:.2f}M"
        flops_g = f"{r['flops'] / 1e9:.2f}G"
        latency_ms = f"{r['latency'] * 1000:.3f}ms"

        print(f"| {model_name} | {method} | {amount_pct} | **{acc_pct}** | {params_m} | {flops_g} | {latency_ms} |")
    print("\n---")


# --- Main Execution ---
def main():
    if not MODEL_CLASSES:
        print("Error: MODEL_CLASSES registry is empty.")
        print("Please edit `benchmarker.py` to import and register your model classes.")
        return

    print(f"--- Starting Benchmark (using {DEVICE}) ---")

    results = []
    criterion = nn.CrossEntropyLoss()

    # --- 1. Benchmark Original Models ---
    print(f"\nScanning for original models in: {ORIGINAL_MODELS_DIR}")
    original_model_files = glob.glob(os.path.join(ORIGINAL_MODELS_DIR, "*_best.pt"))

    for file_path in original_model_files:
        filename = os.path.basename(file_path)
        model_name = filename.replace("_best.pt", "")

        if model_name not in MODEL_CLASSES:
            print(f"Warning: Skipping {filename}, model class '{model_name}' not in registry.")
            continue

        print(f"Benchmarking original model: {model_name}")
        ModelClass = MODEL_CLASSES[model_name]
        model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)

        checkpoint = torch.load(file_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        metrics = run_benchmark(model, val_loader, criterion, DEVICE)
        metrics.update({
            "model": model_name,
            "method": "Original",
            "amount": 0.0
        })
        results.append(metrics)

    # --- 2. Benchmark Pruned Models ---
    print(f"\nScanning for pruned models in: {PRUNED_MODELS_DIR}")
    pruned_model_files = glob.glob(os.path.join(PRUNED_MODELS_DIR, "pruned_*.pt"))

    for file_path in pruned_model_files:
        filename = os.path.basename(file_path)
        print(f"Benchmarking pruned model: {filename}")

        try:
            # Parse filename: pruned_LeNet_unstructured_0.3.pt
            parts = filename.replace(".pt", "").split("_")

            amount = float(parts[-1])
            pruning_type = parts[-2]

            # Connect together the names that were divided by '_' (like SimpleCNN_6x2)
            model_name = "_".join(parts[1:-2])

        except Exception as e:
            print(f"Warning: Skipping {filename}, could not parse name. Error: {e}")
            continue

        if model_name not in MODEL_CLASSES:
            print(f"Warning: Skipping {filename}, model class '{model_name}' not in registry.")
            continue

        ModelClass = MODEL_CLASSES[model_name]
        model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)

        # --- CRITICAL STEP: Apply pruning *before* loading state_dict ---
        if pruning_type == "unstructured":
            apply_unstructured(model, amount)
        elif pruning_type == "structured":
            apply_structured(model, amount)
        else:
            print(f"Warning: Skipping {filename}, unknown pruning type '{pruning_type}'.")
            continue

        # Load and Clean Checkpoint
        checkpoint = torch.load(file_path, map_location=DEVICE)
        state_dict = checkpoint["model_state_dict"]
        keys_to_remove = [k for k in state_dict.keys() if "total_ops" in k or "total_params" in k]
        for k in keys_to_remove: del state_dict[k]

        model.load_state_dict(state_dict, strict=False)


        # --- CRITICAL STEP: Make pruning permanent ---
        make_pruning_permanent(model)

        method = "structured" if "structured" in filename and "unstructured" not in filename else "unstructured"

        if method == "structured":
            input_size = get_input_size(model)
            # Physical removal of channels (requires correct input size)
            model = physically_prune_structured(model, pruning_ratio=amount, example_input_size=input_size)

        try:
            metrics = run_benchmark(model, val_loader, criterion, DEVICE)
            metrics.update({
                "model": model_name,
                "method": pruning_type.capitalize(),
                "amount": amount
            })
            results.append(metrics)
        except Exception as e:
            print(f"Error benchmarking {filename}: {e}")

    # --- 3. Generate Report ---
    if results:
        generate_markdown_report(results)
    else:
        print("No models found or benchmarked. Check your paths and MODEL_CLASSES registry.")


if __name__ == "__main__":
    main()