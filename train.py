"""
DeepForge - Training Script
Trains ContentEncoder + StyleEncoder + Generator (as a combined
"style-transfer network") adversarially against a Discriminator.

Losses used:
  1. Adversarial loss   - generator fools discriminator (BCE)
  2. Reconstruction loss - generated image matches target (L1)
  3. Style loss (Gram)   - generated image's style stats match style image's

Run:
    python train.py --epochs 20 --batch_size 64
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os

from data.emnist_loader import get_dataloaders
from models.content_encoder import ContentEncoder
from models.style_encoder import StyleEncoder
from models.generator import Generator
from models.discriminator import Discriminator
from utils import save_sample_grid, gram_matrix, weights_init


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, _ = get_dataloaders(batch_size=args.batch_size, split=args.split)

    # Models
    content_enc = ContentEncoder().to(device)
    style_enc = StyleEncoder().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Losses
    adv_loss_fn = nn.BCEWithLogitsLoss()
    recon_loss_fn = nn.L1Loss()

    # Optimizers (encoders + generator trained together)
    g_params = list(content_enc.parameters()) + list(style_enc.parameters()) + list(generator.parameters())
    opt_g = optim.Adam(g_params, lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        for i, (content_img, style_img, target_img, _labels) in enumerate(train_loader):
            content_img = content_img.to(device)
            style_img = style_img.to(device)
            target_img = target_img.to(device)
            b = content_img.size(0)

            real_labels = torch.ones((b, 1, 7, 7), device=device)
            fake_labels = torch.zeros((b, 1, 7, 7), device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_d.zero_grad()

            content_feat = content_enc(content_img)
            style_vec = style_enc(style_img)
            fake_img = generator(content_feat, style_vec)

            real_pred = discriminator(target_img)
            fake_pred = discriminator(fake_img.detach())

            d_loss_real = adv_loss_fn(real_pred, real_labels)
            d_loss_fake = adv_loss_fn(fake_pred, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            d_loss.backward()
            opt_d.step()

            # -----------------
            #  Train Generator
            # -----------------
            opt_g.zero_grad()

            fake_pred_for_g = discriminator(fake_img)
            g_adv_loss = adv_loss_fn(fake_pred_for_g, real_labels)
            g_recon_loss = recon_loss_fn(fake_img, target_img)

            # simple style consistency loss via Gram matrices on raw images
            g_style_loss = recon_loss_fn(gram_matrix(fake_img), gram_matrix(style_img))

            g_loss = g_adv_loss + args.lambda_recon * g_recon_loss + args.lambda_style * g_style_loss
            g_loss.backward()
            opt_g.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f} "
                      f"(adv: {g_adv_loss.item():.4f}, recon: {g_recon_loss.item():.4f}, style: {g_style_loss.item():.4f})")

        # Save sample outputs + checkpoint each epoch
        save_sample_grid(fake_img[:16], f"outputs/epoch_{epoch+1}.png")
        torch.save({
            "content_enc": content_enc.state_dict(),
            "style_enc": style_enc.state_dict(),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        }, f"checkpoints/deepforge_epoch{epoch+1}.pt")
        print(f"Saved sample + checkpoint for epoch {epoch+1}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--split", type=str, default="letters")
    parser.add_argument("--lambda_recon", type=float, default=10.0)
    parser.add_argument("--lambda_style", type=float, default=5.0)
    args = parser.parse_args()

    train(args)
