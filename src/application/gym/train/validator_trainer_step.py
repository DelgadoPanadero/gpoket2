from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from src.application.gym.model.sprite_validator import _to_tensor

from src.application.gym.model.sprite_validator import SpriteValidator

_TARGET_SIZE = 64


def _load_png_as_rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[0] != _TARGET_SIZE or img.shape[1] != _TARGET_SIZE:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # Composite RGBA onto white background
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        img = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
    return img  # BGR uint8


def _scramble_rows(img: np.ndarray) -> np.ndarray:
    rows = list(img)
    np.random.shuffle(rows)
    return np.stack(rows)


def _random_noise() -> np.ndarray:
    return np.random.randint(0, 256, (_TARGET_SIZE, _TARGET_SIZE, 3), dtype=np.uint8)


def _pixel_corrupt(img: np.ndarray, frac: float = 0.5) -> np.ndarray:
    out = img.copy()
    mask = np.random.rand(_TARGET_SIZE, _TARGET_SIZE) < frac
    out[mask] = np.random.randint(0, 256, (mask.sum(), 3), dtype=np.uint8)
    return out


class _SpriteDataset(Dataset):
    def __init__(self, positives: list[np.ndarray]):
        self.positives = positives
        # Generate one negative for each positive to keep dataset balanced
        self.negatives = self._make_negatives(positives)
        pass  # transform defined in __getitem__

    def _make_negatives(self, positives: list[np.ndarray]) -> list[np.ndarray]:
        negs: list[np.ndarray] = []
        strategies = [_scramble_rows, _random_noise, _pixel_corrupt]
        for i, img in enumerate(positives):
            fn = strategies[i % len(strategies)]
            negs.append(fn(img) if fn is not _random_noise else _random_noise())
        return negs

    def __len__(self) -> int:
        return len(self.positives) + len(self.negatives)

    def __getitem__(self, idx: int):
        if idx < len(self.positives):
            bgr = self.positives[idx]
            label = 1.0
        else:
            bgr = self.negatives[idx - len(self.positives)]
            label = 0.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _to_tensor(rgb), torch.tensor(label)


class ValidatorTrainerStep:
    def __init__(
        self,
        sprite_root: str | Path = "data/brz/pokemon",
        output_path: str | Path = "data/gld/thinbaker_team/sprite_validator.pt",
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-4,
        val_split: float = 0.1,
    ):
        self.sprite_root = Path(sprite_root)
        self.output_path = Path(output_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_split = val_split

    def _load_sprites(self) -> list[np.ndarray]:
        sprites: list[np.ndarray] = []
        for png in sorted(self.sprite_root.rglob("*.png")):
            img = _load_png_as_rgb(png)
            if img is not None:
                sprites.append(img)
        return sprites

    def run(self) -> Path:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading sprites from {self.sprite_root}…")
        positives = self._load_sprites()
        print(f"Loaded {len(positives)} valid 64×64 sprites")

        dataset = _SpriteDataset(positives)
        val_len = max(1, int(len(dataset) * self.val_split))
        train_len = len(dataset) - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=2)

        model = SpriteValidator().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images)
                    correct += ((preds > 0.5) == labels.bool()).sum().item()
                    total += len(labels)

            acc = correct / total if total else 0.0
            print(
                f"Epoch {epoch}/{self.epochs}  "
                f"loss={train_loss / len(train_loader):.4f}  "
                f"val_acc={acc:.3f}"
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(self.output_path)
        print(f"Validator saved to {self.output_path}")
        return self.output_path
