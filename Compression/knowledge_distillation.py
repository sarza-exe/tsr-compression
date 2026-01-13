import os
import sys

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Architectures.cnn_6x2 import SimpleCNN_6x2
from data_original import train_loader, val_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
TEMP = 6.0  # Temperature for Softmax
ALPHA = 0.7  # 0.7 for Teacher, 0.3 for Hard Labels
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 4. Helper: Loading Pruned Model
# ==========================================
def load_structured_student(checkpoint_path):
    print(f"--- Loading Student from: {checkpoint_path} ---")

    # 1. Initialize the clean model
    model = SimpleCNN_6x2()

    model.to("cpu")

    # 2. Re-apply the pruning structure
    print(f"Applying dummy pruning structure to match checkpoint...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=0.7,
                n=2,
                dim=0
            )

    # 3. Load the state dictionary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded pruned model.")

    except Exception as e:
        print(f"Error loading student model: {e}")
        raise e

    return model


# ==========================================
# 5. Loss Function for Distillation
# ==========================================
def distillation_loss_fn(student_logits, teacher_logits, labels, T, alpha):
    # 1. Distillation Loss (Soft Targets)
    # KLDivLoss expects LogSoftmax as input
    distillation = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)

    # 2. Student Loss (Hard Targets)
    student_loss = F.cross_entropy(student_logits, labels)

    # 3. Weighted Sum
    return alpha * distillation + (1.0 - alpha) * student_loss


# ==========================================
# 6. Main Training Loop
# ==========================================
def main():
    print(f"Running on device: {DEVICE}")

    # --- A. Prepare Teacher ---
    teacher = SimpleCNN_6x2().to(DEVICE)
    # Load your trained teacher weights here
    teacher_checkpoint = torch.load("../Models/SimpleCNN_6x2_best.pt", map_location=DEVICE)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])  # <--- Wyciągamy wagi
    print("Teacher model loaded.")
    teacher.eval()  # Freeze teacher

    # --- B. Prepare Student (Pruned) ---
    # Assuming 'student_pruned.pt' is your file with masks
    student = load_structured_student("../Compressed_Models/pruned_SimpleCNN_6x2_structured_0.7.pt")

    student.to(DEVICE)
    student.train()  # Set to train mode for fine-tuning

    # Optimizer
    optimizer = optim.SGD(student.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # --- Sanity check student BEFORE distillation ---
    student.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = student(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Student accuracy BEFORE KD: {100 * correct / total:.2f}%")
    student.train()


    print("Starting Knowledge Distillation Fine-tuning...")

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward Pass Student
            student_logits = student(inputs)

            # Forward Pass Teacher (No Grad)
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Calculate Loss
            loss = distillation_loss_fn(student_logits, teacher_logits, labels, TEMP, ALPHA)

            # Backward Pass & Optimize
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 50 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i}], Loss: {loss.item():.4f}")

        # Epoch Summary
        epoch_acc = 100 * correct / total
        print(
            f"--- Epoch [{epoch + 1}/{EPOCHS}] Average Loss: {running_loss / len(train_loader):.4f} | Student Acc: {epoch_acc:.2f}% ---")

    # --- C. Save the Distilled Model ---
    # We save it so it can be quantized later
    torch.save(student.state_dict(), "student_distilled.pt")
    print("Model saved as 'student_distilled.pt'. Ready for Quantization.")


if __name__ == "__main__":
    main()