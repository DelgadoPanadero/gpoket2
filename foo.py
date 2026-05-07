import re
import numpy as np
from pathlib import Path
from PIL import Image

_ROW_RE = re.compile(r"^\d{2}$")
_BLANK = "~"
_WHITE = (255, 255, 255)
_IMAGE_WIDTH = 64

_INFERENCE_RE = re.compile(
    r"=== Inference @ step (\d+) ===\n(.*?)\n====================================",
    re.DOTALL,
)


def _char_to_rgb(char: str) -> tuple[int, int, int]:
    if char == _BLANK:
        return _WHITE
    idx = ord(char) - 59
    r, g, b = idx // 16, (idx % 16) // 4, idx % 4
    return (r * 85, g * 85, b * 85)


def _text_to_image(text: str) -> Image.Image:
    rows: list[list[str]] = []
    current: list[str] = []
    for token in text.split():
        if _ROW_RE.match(token):
            if current:
                rows.append(current)
            current = []
        elif len(token) == 1:
            current.append(token)
    if current:
        rows.append(current)

    height = len(rows) or _IMAGE_WIDTH
    img = np.full((height, _IMAGE_WIDTH, 3), 255, dtype=np.uint8)
    for y, row in enumerate(rows):
        for x, char in enumerate(row[: _IMAGE_WIDTH - 1]):
            img[y, x + 1] = _char_to_rgb(char)

    return Image.fromarray(img)


def main():
    log_path = Path("train.log")
    output_dir = Path("data/gld/training_pc")
    output_dir.mkdir(parents=True, exist_ok=True)

    content = log_path.read_text(errors="replace")
    matches = _INFERENCE_RE.findall(content)

    print(f"Found {len(matches)} inference samples in {log_path}")

    for step, text in matches:
        image = _text_to_image(text.strip())
        output_path = output_dir / f"step_{int(step):05d}.png"
        image.save(output_path)
        print(f"  step {step:>5} → {output_path}")


if __name__ == "__main__":
    main()