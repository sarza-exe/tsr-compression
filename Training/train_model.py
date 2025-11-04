import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Architectures.cnn_6x2 import SimpleCNN_6x2
from data_original import train_loader, val_loader
from train_utils import train_one_epoch, validate

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS_DIR = "../Models"
# os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "cnn6x2_best.pt")

NUM_CLASSES = 43           # GTSRB dataset has 43 traffic sign classes
LR = 1e-3                  # Learning rate for optimizer
WEIGHT_DECAY = 1e-4        # Weight decay for L2 regularization
EPOCHS = 30                # Number of epochs to train


def main():

    model = SimpleCNN_6x2(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_acc)

        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f" Val  loss: {val_loss:.4f},  Val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH} (val_acc={val_acc:.4f})")

    print("Training completed.")
    print(f"Best validation accuracy achieved: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()