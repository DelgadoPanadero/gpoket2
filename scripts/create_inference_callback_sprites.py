import re
import cv2
import numpy as np
from pathlib import Path

_BLANK = "~"
_WHITE = (255, 255, 255)
_IMAGE_SIZE = 64
_ROW_PREFIX = "[ROW_"
_SPECIAL_TOKEN_RE = re.compile(
    r"^\[[A-Z_0-9]+\]$",
)  # [BOS], [EOS], [PAD], [UNK]

_INFERENCE_RE = re.compile(
    r"=== Inference @ step (\d+) ===\n(.*?)\n====================================",
    re.DOTALL,
)


def _char_to_rgb(char: str) -> tuple[int, int, int]:
    if char == _BLANK:
        return _WHITE
    idx = ord(char) - 59
    r, g, b = idx // 16, (idx // 4) % 4, idx % 4
    return (r * 64 + 32, g * 64 + 32, b * 64 + 32)


def _text_to_image(text: str) -> np.ndarray:
    rows: list[list[str]] = []
    current: list[str] = []

    for token in text.split():
        if token.startswith(_ROW_PREFIX) and token.endswith("]"):
            if current:
                rows.append(current)
            current = []
        elif _SPECIAL_TOKEN_RE.match(token):
            pass  # skip [BOS], [EOS], [PAD], [UNK]
        elif len(token) == 1:
            current.append(token)
        else:
            # BPE merge token: each char is one pixel
            current.extend(list(token))

    if current:
        rows.append(current)

    img = np.full((_IMAGE_SIZE, _IMAGE_SIZE, 3), 255, dtype=np.uint8)
    for y, row in enumerate(rows[:_IMAGE_SIZE]):
        for x, char in enumerate(row[:_IMAGE_SIZE]):
            r, g, b = _char_to_rgb(char)
            img[y, x] = np.clip((b, g, r), 0, 255)  # cv2 uses BGR

    return img


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
        cv2.imwrite(str(output_path), image)
        print(f"  step {step:>5} → {output_path}")


if __name__ == "__main__":
    main()
