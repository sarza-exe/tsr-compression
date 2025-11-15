import os
import sys
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_original import train_loader, val_loader
from Training.train_utils import train_one_epoch, validate
from metrics_utils import count_params, count_flops, measure_latency


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "../Compressed_Models"
os.makedirs(SAVE_DIR, exist_ok=True)

PRUNING_RATIOS = [0.3, 0.5, 0.7]  # pruning levels to test
FINE_TUNE_EPOCHS = 10
LR = 1e-4


# Unstructured pruning
def apply_unstructured(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model

# Structured pruning
def apply_structured(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    return model


# Run pruning experiment for a given model class
def run_experiment(ModelClass):
    results = []

    for pruning_type in ["unstructured", "structured"]:
        for amount in PRUNING_RATIOS:
            print(f"\n=== Running {pruning_type} pruning, amount={amount} ===")

            # Initialize model and load pretrained weights
            model = ModelClass(num_classes=43).to(DEVICE)
            pretrained_path = f"../Models/{ModelClass.__name__}_best.pt"
            checkpoint = torch.load(pretrained_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])

            # Apply pruning
            if pruning_type == "unstructured":
                model = apply_unstructured(model, amount)
            else:
                model = apply_structured(model, amount)

            # Setup optimizer and loss
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()

            # Fine-tune pruned model
            best_val_acc = 0.0
            for epoch in range(1, FINE_TUNE_EPOCHS + 1):
                train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
                val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
                print(f"Epoch {epoch}, val_acc={val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            # Collect metrics
            params = count_params(model)
            flops = count_flops(model)
            latency = measure_latency(model, device="cpu")

            # Save pruned model
            save_path = os.path.join(SAVE_DIR, f"pruned_{ModelClass.__name__}_{pruning_type}_{amount}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "pruning_type": pruning_type,
                "amount": amount,
                "params": params,
                "flops": flops,
                "latency": latency,
                "val_acc": best_val_acc,
            }, save_path)
            print(f"Saved: {save_path}")

            # Append results to list
            results.append({
                "model": ModelClass.__name__,
                "method": pruning_type,
                "amount": amount,
                "val_acc": best_val_acc,
                "params": params,
                "flops": flops,
                "latency": latency
            })

    return results
