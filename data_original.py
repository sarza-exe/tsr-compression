from torch.utils.data import DataLoader
# from Datasets.sources.GTSRB import GTSRB
from torchvision.datasets import GTSRB
import torchvision.transforms as transform
from Datasets.Augmentation.showAugmentation import showBatch

root = ("data/GTSRB")

# Resize images to 32x32, convert to tensor, and normalize pixel values
train_transforms = transform.Compose([
    transform.Resize((32, 32)),
    transform.ToTensor(),
    transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transforms = transform.Compose([
    transform.Resize((32, 32)),
    transform.ToTensor(),
    transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load the GTSRB training and validation/testing dataset using the defined transforms
train_dataset = GTSRB(root=root, split='train', transform=train_transforms)
val_dataset = GTSRB(root=root, split='test', transform=val_transforms)

# Create DataLoader for the training dataset (shuffled for randomness) and validation dataset (not shuffled)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)


# Example: iterate one batch to sanity-check shapes
if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print("Train batch - images shape:", images.shape)  # expected (Batch_size, No. RGB channels, Height, Width)
    print("Train batch - labels shape:", labels.shape)
    # Check value range after normalization
    print("Images min / max (normalized):", images.min().item(), images.max().item())

    showBatch(train_loader)

