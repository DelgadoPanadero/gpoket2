from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


def _to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8 RGB array to CxHxW float32 tensor in [0, 1]."""
    return torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0


class SpriteValidator(nn.Module):
    """
    Lightweight CNN discriminator (~500K params) that scores a 64×64 RGB image
    as a valid Pokemon sprite (score → 1) or invalid/corrupted (score → 0).

    Trained separately via ValidatorTrainerStep on:
      Positives — real sprites from the bronze dataset
      Negatives — scrambled rows, random noise, pixel-corrupted sprites
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64→32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32→16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16→8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8→4
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 64, 64) float in [0, 1]. Returns (B,) scores."""
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def score(self, bgr_image: np.ndarray) -> float:
        """Score a single 64×64 BGR numpy array (OpenCV format). Returns float in [0,1]."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = _to_tensor(rgb).unsqueeze(0).to(next(self.parameters()).device)
        return self.forward(tensor).item()

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SpriteValidator":
        model = cls()
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True),
        )
        model.to(device)
        model.eval()
        return model
