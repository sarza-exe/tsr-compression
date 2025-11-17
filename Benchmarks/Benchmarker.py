import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Model architectures:
from Architectures.cnn_6x2 import SimpleCNN_6x2
from Architectures.efficientnet_b0_custom import EfficientNetB0Custom
from Architectures.enhanced_lenet5 import EnhancedLeNet5
from Architectures.resnet50_custom import ResNet50Custom


# --- 2. USER: Import Your Helper Functions ---
from data_original import val_loader  # Assumes this has the GTSRB val_loader
from Training.train_utils import validate
from Compression.metrics_utils import count_params, count_flops, measure_latency

# --- 3. USER: Configure Your Model Registry ---
# Match the class name (string) to the imported class
MODEL_CLASSES = {
    # "YourModelClass1": YourModelClass1,
    # "YourModelClass2": YourModelClass2,
    # "LeNet": LeNet, # Example
    "CNN_6x2": SimpleCNN_6x2,
    "EfficientNetB0": EfficientNetB0Custom,
    "EnhancedLeNet5" : EnhancedLeNet5,
    "ResNet50": ResNet50Custom,
}

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORIGINAL_MODELS_DIR = "../Models"
PRUNED_MODELS_DIR = "../Compressed_Models"
NUM_CLASSES = 43  # For GTSRB


# --- Pruning helper functions (copied from your script) ---
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
    # Note: count_params should count *non-zero* params
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
            model_name = parts[1]
            pruning_type = parts[2]
            amount = float(parts[3])
        except Exception as e:
            print(f"Warning: Skipping {filename}, could not parse name. Error: {e}")
            continue

        if model_name not in MODEL_CLASSES:
            print(f"Warning: Skipping {filename}, model class '{model_name}' not in registry.")
            continue

        ModelClass = MODEL_CLASSES[model_name]
        model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)

        # --- CRITICAL STEP: Apply pruning *before* loading state_dict ---
        # This re-parameterizes the model to have 'weight_orig' and 'weight_mask'
        if pruning_type == "unstructured":
            apply_unstructured(model, amount)
        elif pruning_type == "structured":
            apply_structured(model, amount)
        else:
            print(f"Warning: Skipping {filename}, unknown pruning type '{pruning_type}'.")
            continue

        # Load the saved state dict (weights and masks)
        checkpoint = torch.load(file_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        # --- CRITICAL STEP: Make pruning permanent ---
        # This removes the pruning "hooks" and bakes the 0s into the weights
        # This is essential for count_params and latency tests
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if prune.is_pruned(module):
                    prune.remove(module, 'weight')

        # Run benchmarks on the finalized pruned model
        metrics = run_benchmark(model, val_loader, criterion, DEVICE)
        metrics.update({
            "model": model_name,
            "method": pruning_type.capitalize(),
            "amount": amount
        })
        results.append(metrics)

    # --- 3. Generate Report ---
    if results:
        generate_markdown_report(results)
    else:
        print("No models found or benchmarked. Check your paths and MODEL_CLASSES registry.")


if __name__ == "__main__":
    main()