import os
import sys
import torch
import torch.nn as nn
import re
import csv
from typing import Dict, Type

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Architecture Imports ---
from Architectures.cnn_6x2 import SimpleCNN_6x2
from Architectures.enhanced_lenet5 import EnhancedLeNet5
from Architectures.resnet50_custom import ResNet50Custom
from Architectures.efficientnet_b0_custom import EfficientNetB0Custom
from metrics_utils import count_params, count_flops, measure_latency, get_input_size
from slimming_utils import physically_prune_structured, make_pruning_permanent

# --- Configuration ---
INPUT_DIR: str = "../Compressed_Models/Pruned_normal"
OUTPUT_DIR: str = "../Compressed_Models/Pruned_slimmed"
CSV_FILE: str = "../Results/pruning_results.csv"
SAVE_SUFFIX: str = "_slimmed"
DEVICE: str = "cpu"
NUM_CLASSES: int = 43

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# Model Mapping
MODEL_MAPPING: Dict[str, Type[nn.Module]] = {
    "SimpleCNN_6x2": SimpleCNN_6x2,
    "EnhancedLeNet5": EnhancedLeNet5,
    "ResNet50Custom": ResNet50Custom,
    "EfficientNetB0Custom": EfficientNetB0Custom
}



# --- MAIN EXECUTION ---

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    # 1. Initialize CSV file
    file_exists = os.path.isfile(CSV_FILE)
    csv_file = open(CSV_FILE, mode='a', newline='')
    writer = csv.writer(csv_file)

    if not file_exists:
        headers = ['model', 'method', 'amount', 'val_acc', 'params', 'flops', 'latency']
        writer.writerow(headers)
        print(f"Created CSV: {CSV_FILE}")

    # 2. Find files
    files = [f for f in os.listdir(INPUT_DIR)
             if f.endswith(".pt") and "pruned_" in f and os.path.isfile(os.path.join(INPUT_DIR, f))]
    files.sort()

    print(f"Found {len(files)} models to process in {INPUT_DIR}.\n")

    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)

        try:
            # --- Parse Filename Info ---
            model_name = next((name for name in MODEL_MAPPING if name in filename), None)
            if not model_name: continue

            method = "structured" if "structured" in filename and "unstructured" not in filename else "unstructured"
            ratio_match = re.findall(r"0\.\d+", filename)
            amount = float(ratio_match[-1]) if ratio_match else 0.0

            print(f"Processing: {model_name} | {method} | {amount}")

            # --- Load Model ---
            ModelClass = MODEL_MAPPING[model_name]
            model = ModelClass(num_classes=NUM_CLASSES)

            # Retrieve correct input size for this model
            input_size = get_input_size(model)

            checkpoint = torch.load(filepath, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            val_acc = checkpoint.get('val_acc', 0.0)
            if isinstance(val_acc, torch.Tensor): val_acc = val_acc.item()

            model.load_state_dict(state_dict, strict=False)

            # --- Clean / Slim Model ---
            if method == "structured":
                # Physical removal of channels (requires correct input size)
                model = physically_prune_structured(model, pruning_ratio=amount, example_input_size=input_size)
            else:
                # Making zero weights permanent
                model = make_pruning_permanent(model)

            # --- Calculate Metrics (using internal functions) ---
            model.to(DEVICE)

            params = count_params(model)
            flops = count_flops(model, device=DEVICE)
            latency = measure_latency(model, device=DEVICE)

            # --- Save Results ---
            row = [model_name, method, amount, f"{val_acc:.4f}", params, flops, f"{latency:.6f}"]
            writer.writerow(row)
            csv_file.flush()

            # --- Save Cleaned .pt File ---
            new_filename = filename.replace(".pt", f"{SAVE_SUFFIX}.pt")
            save_path = os.path.join(OUTPUT_DIR, new_filename)

            torch.save({
                'model_state_dict': model.state_dict(),
                'params': params,
                'flops': flops,
                'latency': latency,
                'val_acc': val_acc
            }, save_path)

            print(f"   Saved to: {save_path}")
            print(f"   Params: {params / 1e6:.2f}M | Latency: {latency * 1000:.2f}ms")

        except Exception as e:
            print(f"    Error processing {filename}: {e}")

    csv_file.close()
    print(f"\nProcessing finished. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
