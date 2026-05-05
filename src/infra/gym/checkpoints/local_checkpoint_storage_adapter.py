import os
import re
import shutil
from pathlib import Path


class LocalCheckpointStorageAdapter:
    layer = "train"
    entity_name = "checkpoints"

    def __init__(
        self,
        base_path: Path | str = Path("./data/"),
        dataset: str = "latest",
        experiment: str = "",
    ):

        if dataset == "":
            if all_dataset := sorted(
                [
                    dir_name.name
                    for dir_name in (
                        Path(base_path) / self.layer / self.entity_name
                    ).glob("box-*")
                    if dir_name.is_dir()
                ],
            ):
                dataset = all_dataset[-1]

        self.base_path = (
            Path(base_path)
            / self.layer
            / self.entity_name
            / dataset
            / experiment
        )
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_latest_checkpoint(
        self,
    ) -> str | None:
        checkpoints = []
        for file_name in os.listdir(self.base_path):
            if match := re.match(r"checkpoint-(\d+)", file_name):
                checkpoints.append(
                    {
                        "step": int(match.group(1)),
                        "checkpoint_path": os.path.join(
                            self.base_path,
                            file_name,
                        ),
                    },
                )

        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: x["step"], reverse=True)
        for checkpoint in checkpoints:
            state_file = os.path.join(
                checkpoint["checkpoint_path"], "trainer_state.json"
            )
            if os.path.exists(state_file) and os.path.getsize(state_file) > 0:
                return checkpoint["checkpoint_path"]

        return None

    def save_checkpoint(
        self,
        trainer_checkpoint_path: str,
    ):
        shutil.move(
            src=trainer_checkpoint_path,
            dst=os.path.join(
                self.base_path,
                os.path.basename(trainer_checkpoint_path),
            ),
        )

    def load_checkpoint(
        self,
        checkpoint_path: str,
        trainer_checkpoint_dir: str,
    ):
        shutil.copytree(
            src=checkpoint_path,
            dst=os.path.join(
                trainer_checkpoint_dir,
                os.path.basename(checkpoint_path),
            ),
        )
