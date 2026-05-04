import re
from pathlib import Path


class LocalCheckpointStorageRepository:
    def __init__(
        self,
        base_dir: Path | str = Path("/workspace/train/checkpoints/"),
        dataset: str = "latest",
        experiment: str = "",
    ):
        self.base_dir = Path(base_dir) / dataset / experiment

    def get_latest_checkpoint(self) -> Path:
        checkpoints = sorted(
            [d for d in self.base_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda d: int(re.search(r"\d+", d.name).group()),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.base_dir}")
        return checkpoints[-1]
