"""
Utility helpers for DeepForge.
"""

import torch
import torchvision.utils as vutils
import os


def save_sample_grid(images, path, nrow=8):
    """Save a batch of images (in [-1,1]) as a single grid PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = (images + 1) / 2  # rescale [-1,1] -> [0,1]
    vutils.save_image(images, path, nrow=nrow)


def gram_matrix(feat):
    """
    Compute Gram matrix for style loss (classic Neural Style Transfer trick).
    feat: (B, C, H, W)
    """
    b, c, h, w = feat.size()
    features = feat.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def weights_init(m):
    """Standard DCGAN-style weight initialization."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "InstanceNorm" in classname or "BatchNorm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
