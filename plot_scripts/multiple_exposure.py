from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def create_multiple_exposure(
    folder: str | Path,
    output_path: str | Path | None = None,
    prominence_power: float = 2.0,
    min_opacity: float = 0.35,
) -> Path:
    """Overlay changed pixels from lexicographically sorted PNG frames onto a fixed background."""
    folder = Path(folder)
    frame_paths = sorted(folder.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG files found in {folder}")

    if prominence_power <= 0:
        raise ValueError("prominence_power must be positive")
    if not 0.0 <= min_opacity <= 1.0:
        raise ValueError("min_opacity must be between 0 and 1")

    weights = np.arange(1, len(frame_paths) + 1, dtype=np.float32) ** prominence_power
    opacities = weights / weights[-1]
    opacities = min_opacity + (1.0 - min_opacity) * opacities

    with Image.open(frame_paths[0]) as first_image:
        background = np.asarray(first_image.convert("RGBA"), dtype=np.float32)
        width, height = background.shape[1], background.shape[0]
        output = background.copy()

    for frame_path, opacity in zip(frame_paths[1:], opacities[1:]):
        with Image.open(frame_path) as image:
            rgba = image.convert("RGBA")
            if rgba.size != (width, height):
                raise ValueError(
                    f"Frame size mismatch for {frame_path}: expected {(width, height)}, got {rgba.size}"
                )
            frame = np.asarray(rgba, dtype=np.float32)

        changed = np.any(frame != background, axis=-1, keepdims=True)
        if not np.any(changed):
            continue

        blended = output * (1.0 - opacity) + frame * opacity
        output = np.where(changed, blended, output)

    output_path = Path(output_path) if output_path is not None else folder / "multiple_exposure.png"
    Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGBA").save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a multiple-exposure image from PNG frames.")
    parser.add_argument("folder", help="Folder containing PNG frames.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output image path. Defaults to <folder>/multiple_exposure.png.",
    )
    parser.add_argument(
        "--prominence-power",
        type=float,
        default=2.0,
        help="Higher values make later frames more prominent. Default: 2.0.",
    )
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.35,
        help="Opacity used for the earliest visible trail frames. Default: 0.35.",
    )
    args = parser.parse_args()
    create_multiple_exposure(args.folder, args.output, args.prominence_power, args.min_opacity)


if __name__ == "__main__":
    main()
