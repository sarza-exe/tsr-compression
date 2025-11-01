import torch
import matplotlib.pyplot as plt

def unnormalize(img_tensor, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    """
    Undo normalization: img_tensor is (C,H,W) with normalization applied.
    Returns a numpy array HxWxC in range [0,1].
    """
    # Move to cpu and clone to avoid in-place ops
    img = img_tensor.detach().cpu().clone()
    # Undo normalization: img = img * std + mean
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = torch.clamp(img, 0.0, 1.0)
    # convert to HWC numpy
    return img.permute(1, 2, 0).numpy()

# remove shuffle=True form train_loader in data_augment.py / data_original.py
# to see augmentation of the same image
def showBatch(train_loader):
    # get first batch
    images, labels = next(iter(train_loader))  # images shape: (B, C, H, W)
    n = 8
    images = images[:n]
    labels = labels[:n]

    # plot 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(n):
        img_np = unnormalize(images[i])
        axes[i].imshow(img_np)  # expect HxWxC float in [0,1]
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()