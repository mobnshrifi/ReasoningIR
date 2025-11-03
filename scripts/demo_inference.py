"""Minimal example showing how to run the recurrent NAFNet."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from reasoning_ir import RecurrentNAFNet


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    return to_tensor(image).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.clamp(0, 1)
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.squeeze(0))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the degraded input image")
    parser.add_argument("output", type=Path, help="Where to save the restored output")
    parser.add_argument("--iterations", type=int, default=None, help="Override iterations")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on",
    )
    args = parser.parse_args()

    degraded = load_image(args.image).to(args.device)

    model = RecurrentNAFNet()
    model.to(args.device)
    model.eval()

    with torch.inference_mode():
        restored = model(degraded, num_iterations=args.iterations)

    save_image(restored.cpu(), args.output)
    print(f"Saved restored image to {args.output}")


if __name__ == "__main__":
    main()
