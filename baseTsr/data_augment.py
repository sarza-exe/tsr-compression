import torch
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB
import torchvision.transforms as transform
from showAugmentation import showBatch

# Custom transform to add Gaussian noise to a tensor image
class AddGaussianNoise(object):
    """
    Add Gaussian noise to a tensor image.
    Expects input in range [0, 1] and tensor shape (C, H, W).
    """
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # tensor: torch.Tensor with values in [0,1]
        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy = tensor + noise
        # clamp to valid range
        return torch.clamp(noisy, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

root = "../data/GTSRB" # Root path for GTSRB

# DATA AUGMENTATION
train_transforms = transform.Compose([
    # RandomResizedCrop simulates scaling and random cropping to desired size.
    # scale=(0.8, 1.2) allows slight zoom out/in.
    transform.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.2), ratio=(0.9, 1.1)),

    # RandomAffine for rotation and translation (+/-20 degrees)
    # translate=(0.1,0.1) allows up to 10% shift horizontally/vertically.
    transform.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=0),

    # ColorJitter modifies brightness and contrast
    transform.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.05),

    # Gaussian blur to simulate defocus / motion blur (kernel_size should be odd).
    # sigma range gives variable blur strength.
    transform.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),

    # Convert PIL Image -> Tensor with values in [0,1]
    transform.ToTensor(),

    # Add Gaussian noise to the tensor (after ToTensor).
    AddGaussianNoise(mean=0.0, std=0.03),

    # Normalize to approximately zero-mean, unit-variance.
    transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# VALIDATION / TEST transforms (no augmentation)
val_transforms = transform.Compose([
    transform.Resize((32, 32)),
    transform.ToTensor(),
    transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Create datasets, Note: download=True will fetch the dataset if not present.
train_dataset = GTSRB(root=root, split='train', transform=train_transforms, download=False)
val_dataset = GTSRB(root=root, split='test', transform=val_transforms, download=False)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4) #pin_memory=True
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)


# Example: iterate one batch to sanity-check shapes
if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print("Train batch - images shape:", images.shape)  # expected (Batch_size, No. RGB channels, Height, Width)
    print("Train batch - labels shape:", labels.shape)
    # Check value range after normalization
    print("Images min / max (normalized):", images.min().item(), images.max().item())

    showBatch(train_loader)



