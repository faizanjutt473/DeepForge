"""
DeepForge - Inference Script
Loads a trained checkpoint and generates a handwritten-style image
given a content image (what to write) and a style image (how to write it).

Run:
    python inference.py --checkpoint checkpoints/deepforge_epoch20.pt \
                         --content_idx 5 --style_idx 42
"""

import argparse
import torch

from data.emnist_loader import EMNISTStyleDataset
from models.content_encoder import ContentEncoder
from models.style_encoder import StyleEncoder
from models.generator import Generator
from utils import save_sample_grid


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EMNISTStyleDataset(split=args.split, train=False)

    content_enc = ContentEncoder().to(device)
    style_enc = StyleEncoder().to(device)
    generator = Generator().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    content_enc.load_state_dict(checkpoint["content_enc"])
    style_enc.load_state_dict(checkpoint["style_enc"])
    generator.load_state_dict(checkpoint["generator"])

    content_enc.eval()
    style_enc.eval()
    generator.eval()

    content_img, _, _, _ = dataset[args.content_idx]
    _, style_img, _, _ = dataset[args.style_idx]

    content_img = content_img.unsqueeze(0).to(device)
    style_img = style_img.unsqueeze(0).to(device)

    with torch.no_grad():
        content_feat = content_enc(content_img)
        style_vec = style_enc(style_img)
        generated = generator(content_feat, style_vec)

    # Save content, style, and generated result side-by-side
    grid = torch.cat([content_img, style_img, generated], dim=0)
    save_sample_grid(grid, args.output, nrow=3)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="letters")
    parser.add_argument("--content_idx", type=int, default=0)
    parser.add_argument("--style_idx", type=int, default=1)
    parser.add_argument("--output", type=str, default="outputs/inference_result.png")
    args = parser.parse_args()

    run_inference(args)
