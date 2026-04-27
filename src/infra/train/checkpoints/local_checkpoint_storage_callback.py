import os
import re
import shutil
from pathlib import Path

from transformers import TrainerState  # type: ignore
from transformers import TrainerControl  # type: ignore
from transformers import TrainerCallback  # type: ignore
from transformers import TrainingArguments  # type: ignore


class LocalCheckpointStorageCallback(TrainerCallback):
    def __init__(
        self,
        dataset: str = "",
        base_dir: Path | str = Path(
            "/workspace/GPokeT2/data/train/checkpoints/",
        ),
        experiment: str = "",
    ):
        if dataset == "":
            if all_dataset := sorted(
                [
                    dir_name.name
                    for dir_name in Path(
                        "/workspace/GPokeT2/dataDDD/gld/prof_oak_pc",
                    ).glob("box-*")
                    if dir_name.is_dir()
                ],
            ):
                dataset = all_dataset[-1]

        self.resume_from_checkpoint = None
        self.base_dir = Path(base_dir) / Path(dataset) / Path(experiment)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_latest_checkpoint(
        self,
    ) -> str | None:
        checkpoints = []
        for file_name in os.listdir(self.base_dir):
            if match := re.match(r"checkpoint-(\d+)", file_name):
                checkpoints.append(
                    {
                        "step": int(match.group(1)),
                        "checkpoint_path": os.path.join(
                            self.base_dir,
                            file_name,
                        ),
                    },
                )

        last_checkpoint_path = None
        if checkpoints:
            checkpoints.sort(key=lambda x: x["step"])
            last_checkpoint_path = checkpoints[-1]["checkpoint_path"]

        return last_checkpoint_path

    def _save_checkpoint(
        self,
        trainer_checkpoint_path: str,
    ):
        shutil.move(
            src=trainer_checkpoint_path,
            dst=os.path.join(
                self.base_dir,
                os.path.basename(trainer_checkpoint_path),
            ),
        )

    def _load_checkpoint(
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

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        last_checkpoint_path = self._get_latest_checkpoint()
        if last_checkpoint_path and args.output_dir:
            self._load_checkpoint(
                checkpoint_path=last_checkpoint_path,
                trainer_checkpoint_dir=args.output_dir,
            )

            self.resume_from_checkpoint = os.path.join(
                args.output_dir,
                os.path.basename(last_checkpoint_path),
            )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero and args.output_dir:
            self._save_checkpoint(
                os.path.join(
                    args.output_dir,
                    f"checkpoint-{state.global_step}",
                ),
            )
