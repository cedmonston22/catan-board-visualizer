from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SOURCES = [("test01_jpg.jpg", (0.0, 1.0)), ("test02.jpeg", (0.08, 0.92))]
SOURCE_DIR = Path("datasets/test_photos")
OUTPUT = Path("assets/examples")
LONG_SIDE = 1600
PAD = 0.05


def crop_rows(image: Image.Image, top: float, bottom: float) -> Image.Image:
    pixels = np.array(image)
    height = pixels.shape[0]
    pad = round(PAD * height)
    kept = pixels[round(top * height):round(bottom * height)]
    above = np.full((pad if top > 0 else 0, *kept.shape[1:]), np.median(kept[:20], axis=(0, 1)), dtype=np.uint8)
    below = np.full((pad if bottom < 1 else 0, *kept.shape[1:]), np.median(kept[-20:], axis=(0, 1)), dtype=np.uint8)
    return Image.fromarray(np.vstack([above, kept, below]))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT.glob("*.jpg"):
        old.unlink()
    for i, (name, (top, bottom)) in enumerate(SOURCES, start=1):
        source = SOURCE_DIR / name
        if not source.is_file():
            print(f"no photo at {source}")
            continue
        image = crop_rows(ImageOps.exif_transpose(Image.open(source)).convert("RGB"), top, bottom)
        image.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
        target = OUTPUT / f"board_{i}.jpg"
        image.save(target, quality=90)
        print(f"{source} -> {target} {image.size}, {target.stat().st_size // 1024} KB, EXIF stripped")


if __name__ == "__main__":
    main()
