# DeepForge — Handwriting Style Transfer (Pure PyTorch)

Given a "content" glyph (what to write) and a "style" sample (how someone
writes), DeepForge generates a new handwritten-looking image that combines
the two — built entirely with `torch` / `torchvision` (no external CV or
ML libraries).

## How it works
1. **ContentEncoder** — encodes the shape/structure of the letter to write.
2. **StyleEncoder** — encodes stroke style (thickness, slant, roundness) from a reference sample, pooled into a single style vector.
3. **Generator** — injects the style vector into the content features using **AdaIN** (Adaptive Instance Normalization), then decodes to a 28x28 image.
4. **Discriminator** — a small PatchGAN that pushes the generator to produce realistic strokes.

Training combines three losses:
- Adversarial loss (realism)
- L1 reconstruction loss (matches ground-truth target)
- Gram-matrix style loss (classic Neural Style Transfer trick, matches stroke texture)

## Dataset
Uses `torchvision.datasets.EMNIST` (`split="letters"`) — downloads
automatically, no manual registration needed (this replaces the harder-to-get
IAM Handwriting Database).

## Setup
```bash
pip install torch torchvision
```

## Train
```bash
python train.py --epochs 20 --batch_size 64
```
Sample grids are saved to `outputs/epoch_N.png` after every epoch;
checkpoints are saved to `checkpoints/`.

## Run inference (after training)
```bash
python inference.py --checkpoint checkpoints/deepforge_epoch20.pt \
                     --content_idx 5 --style_idx 42
```
This saves a side-by-side grid: `[content | style | generated]` to
`outputs/inference_result.png`.

## Project structure
```
DeepForge/
├── data/
│   └── emnist_loader.py     # builds (content, style, target) triplets from EMNIST
├── models/
│   ├── content_encoder.py
│   ├── style_encoder.py
│   ├── generator.py         # AdaIN-based style injection
│   └── discriminator.py
├── train.py
├── inference.py
├── utils.py
└── outputs/
```

## Ideas for extending (to make it stand out further)
- Swap EMNIST for a small self-collected dataset of real handwriting for a more convincing demo.
- Add a **few-shot** style mode: average style vectors from 3-5 samples of the same writer.
- Add a Streamlit/Gradio demo where a user types text and picks a style sample.
- Log training with TensorBoard (`torch.utils.tensorboard`) for loss curves — nice for your report/presentation.
