import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Import dataloaders prepared in a separate script
from data_original import train_loader, val_loader


# Set a fixed random seed so that the model training results are reproducible
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

# Automatically select GPU if available, otherwise fall back to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define directories where checkpoints (saved models) will be stored
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Path for saving the best model weights
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cnn6x2_best.pt")

# Define model configuration
NUM_CLASSES = 43           # GTSRB dataset has 43 traffic sign classes
LR = 1e-3                  # Learning rate for optimizer
WEIGHT_DECAY = 1e-4        # Weight decay for L2 regularization
EPOCHS = 30                # Number of epochs to train


# -----------------------
# Model definition — CNN 6+2 architecture
# -----------------------
# This model consists of 6 convolutional layers followed by 2 fully connected layers.
class SimpleCNN_6x2(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()

        # First two convolutional layers: extract low-level features (edges, shapes)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Next two: extract mid-level features (textures, corners)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Final two: extract high-level semantic features
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # Max pooling reduces spatial size, dropout helps prevent overfitting
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # After three poolings, image size reduces: 32x32 → 16x16 → 8x8 → 4x4
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Pass through convolutional layers with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # first pooling

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # second pooling

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)  # third pooling

        # Flatten feature maps into a single vector for fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # apply dropout only during training
        x = self.fc2(x)
        return x


# -----------------------
# Model setup — optimizer, loss, scheduler
# -----------------------
model = SimpleCNN_6x2(num_classes=NUM_CLASSES).to(DEVICE)

# CrossEntropyLoss is used for multi-class classification problems
criterion = nn.CrossEntropyLoss()

# Adam optimizer provides adaptive learning rates for each parameter
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Scheduler reduces learning rate when validation accuracy stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)


# -----------------------
# Training and validation functions
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train the model for one epoch on the training dataset."""
    model.train()  # enable training mode (activates dropout, etc.)
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        # Move data to the selected device (CPU/GPU)
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients from the previous iteration
        optimizer.zero_grad()

        # Forward pass: compute predictions
        outputs = model(images)

        # Compute the loss function
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Track training loss and accuracy
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Return average loss and accuracy for this epoch
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Evaluate the model on the validation dataset without gradient updates."""
    model.eval()  # evaluation mode (disables dropout, etc.)
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # disable gradient computation for efficiency
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Return average loss and accuracy
    return running_loss / total, correct / total


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # Train the model for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate on the test/validation dataset
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        # Adjust learning rate based on validation accuracy
        scheduler.step(val_acc)

        # Display training and validation results
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f" Val  loss: {val_loss:.4f},  Val acc: {val_acc:.4f}")

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, CHECKPOINT_PATH)
            print(f"Saved best model to {CHECKPOINT_PATH} (val_acc={val_acc:.4f})")

    print("Training completed.")
    print(f"Best validation accuracy achieved: {best_val_acc:.4f}")