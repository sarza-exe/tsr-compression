import os
import sys
import torch
import torch.nn as nn

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports
from Architectures.cnn_6x2 import SimpleCNN_6x2
from Architectures.enhanced_lenet5 import EnhancedLeNet5
from Architectures.resnet50_custom import ResNet50Custom
from Architectures.efficientnet_b0_custom import EfficientNetB0Custom

from data_original import val_loader
from Training.train_utils import validate
from Compression.metrics_utils import count_params, measure_latency, get_file_size_mb
from benchmark_utils import load_quantized_model, load_pruned_model, generate_final_report

# Configuration
DEVICE = "cpu"
ORIGINAL_MODELS_DIR = "../Models"
PRUNED_DIR = "../Compressed_Models/Pruned_normal"
SLIMMED_DIR = "../Compressed_Models/Pruned_slimmed"
QUANTIZED_DIR = "../Compressed_Models/Quantization"

os.makedirs(SLIMMED_DIR, exist_ok=True)


MODEL_CLASSES = {
    "SimpleCNN_6x2": SimpleCNN_6x2,
    "EnhancedLeNet5": EnhancedLeNet5,
    "ResNet50Custom": ResNet50Custom,
    "EfficientNetB0Custom": EfficientNetB0Custom,
}


def main():
    all_results = []
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Unified Benchmark (Device: {DEVICE})")

    # 1. Scan Original Models
    print(f"\n>>> Scanning Original Models in {ORIGINAL_MODELS_DIR}")
    if os.path.exists(ORIGINAL_MODELS_DIR):
        for f in os.listdir(ORIGINAL_MODELS_DIR):
            if not f.endswith("_best.pt"): continue

            model_name = f.replace("_best.pt", "")
            if model_name not in MODEL_CLASSES: continue

            print(f"\n-> Benchmarking Original: {model_name}")
            try:
                path = os.path.join(ORIGINAL_MODELS_DIR, f)
                ModelClass = MODEL_CLASSES[model_name]
                model = ModelClass(num_classes=43)
                checkpoint = torch.load(path, map_location="cpu")
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(DEVICE)

                _, acc = validate(model, val_loader, criterion, DEVICE)
                params = count_params(model)
                size_mb = get_file_size_mb(path)
                lat = measure_latency(model, device=DEVICE) * 1000

                all_results.append({
                    "Model": model_name,
                    "Method": "**Original**",
                    "Compression": "N/A",
                    "Val. accuracy": acc * 100,
                    "Num of params (M)": f"{params / 1e6:.2f}",
                    "Size (MB)": f"{size_mb:.2f}",
                    "Latency (ms)": f"{lat:.2f}"
                })
            except Exception as e:
                print(f"\n      Error: {e}")


    # 2. Scan Pruned Models (and save slimmed)
    print(f"\n>>> Scanning Pruned Models in {PRUNED_DIR}")
    if os.path.exists(PRUNED_DIR):
        for f in os.listdir(PRUNED_DIR):
            if not f.endswith(".pt") or "pruned_" not in f: continue

            try:
                # Parse: pruned_SimpleCNN_6x2_structured_0.5.pt
                parts = f.replace(".pt", "").split("_")
                amount = float(parts[-1])
                method = parts[-2]  # structured / unstructured
                model_name = "_".join(parts[1:-2])

                if model_name not in MODEL_CLASSES: continue

                print(f"\n-> Benchmarking Pruned: {model_name} | {method} | {amount}")

                # Load and Slim (in memory)
                model = load_pruned_model(MODEL_CLASSES[model_name], method, amount, os.path.join(PRUNED_DIR, f))
                model.to(DEVICE)

                # save slimmed model
                slimmed_filename = f"{model_name}_{method}_{amount}_slimmed.pt"
                slimmed_path = os.path.join(SLIMMED_DIR, slimmed_filename)
                torch.save(model.state_dict(), slimmed_path)

                # metrics
                _, acc = validate(model, val_loader, criterion, DEVICE)
                params = count_params(model)
                size_mb = get_file_size_mb(slimmed_path)

                lat = measure_latency(model, device=DEVICE) * 1000

                all_results.append({
                    "Model": model_name,
                    "Method": f"Pruning ({method.capitalize()})",
                    "Compression": f"{amount * 100:.0f}%",
                    "Val. accuracy": acc * 100,
                    "Num of params (M)": f"{params / 1e6:.2f}",
                    "Size (MB)": f"{size_mb:.2f}",
                    "Latency (ms)": f"{lat:.2f}"
                })
            except Exception as e:
                print(f"\n      Error: {e}")


    # 3. Scan Quantized Models
    print(f"\n>>> Scanning Quantized Models in {QUANTIZED_DIR}")
    if os.path.exists(QUANTIZED_DIR):
        for f in os.listdir(QUANTIZED_DIR):
            if not f.endswith(".pt") or "quantized_" not in f: continue

            try:
                path = os.path.join(QUANTIZED_DIR, f)
                parts = f.replace(".pt", "").split("_")
                method = parts[-1]
                model_name = "_".join(parts[1:-1])

                if model_name not in MODEL_CLASSES: continue

                print(f"\n-> Benchmarking Quantized: {model_name} | {method}")

                model = load_quantized_model(MODEL_CLASSES[model_name], method, path)
                model.to(DEVICE)

                _, acc = validate(model, val_loader, criterion, DEVICE)

                raw_params = count_params(model)
                params_str = f"{raw_params / 1e6:.2f}" if raw_params > 0 else "N/A"
                size_mb = get_file_size_mb(path)
                lat = measure_latency(model, device=DEVICE) * 1000

                all_results.append({
                    "Model": model_name,
                    "Method": f"Quantization ({method})",
                    "Compression": "INT8",
                    "Val. accuracy": acc * 100,
                    "Num of params (M)": params_str,
                    "Size (MB)": f"{size_mb:.2f}",
                    "Latency (ms)": f"{lat:.2f}"
                })
            except Exception as e:
                print(f"\n      Error: {e}")


    # 4. Generate Report
    if all_results:
        generate_final_report(all_results)
    else:
        print("No results found.")


if __name__ == "__main__":
    main()