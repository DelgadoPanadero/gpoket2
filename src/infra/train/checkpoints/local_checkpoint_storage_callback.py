import os
import re
import shutil
from transformers import TrainerState  # type: ignore
from transformers import TrainerControl  # type: ignore
from transformers import TrainerCallback  # type: ignore
from transformers import TrainingArguments  # type: ignore


class LocalCheckpointStorageCallback(TrainerCallback):

    def __init__(
        self,
        dataset_name: str,
    ):
        self._previous_last_step = 0
        self.backup_dir = f"/home/data/train/checkpoints/{dataset_name}"
        os.makedirs(self.backup_dir, exist_ok=True)

    def _get_latest_checkpoint(
        self,
    ) -> str | None:

        checkpoints = []
        for file_name in os.listdir(self.backup_dir):
            if match := re.match(r"checkpoint-(\d+)", file_name):
                checkpoints.append(
                    {
                        "step": int(match.group(1)),
                        "checkpoint_path": os.path.join(
                            self.backup_dir,
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
        step: int,
    ):
        shutil.move(
            src=trainer_checkpoint_path,
            dst=os.path.join(self.backup_dir, f"checkpoint-{step}"),
        )

    def _load_checkpoint(
        self,
        checkpoint_path: str,
        trainer_checkpoint_dir: str,
    ):
        shutil.copytree(
            src=checkpoint_path,
            dst=trainer_checkpoint_dir,
            dirs_exist_ok=True,
        )

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):

        last_checkpoint_prefix = self._get_latest_checkpoint()
        if last_checkpoint_prefix and args.output_dir:

            self._load_checkpoint(
                checkpoint_path=last_checkpoint_prefix,
                trainer_checkpoint_dir=args.output_dir,
            )

            self._previous_last_step = int(
                last_checkpoint_prefix.split("-")[-1]
            )
            args.resume_from_checkpoint = args.output_dir

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):

        if state.is_world_process_zero and args.output_dir:

            step = state.global_step + self._previous_last_step

            trainer_checkpoint_path = os.path.join(
                args.output_dir,
                f"checkpoint-{state.global_step}",
            )

            self._save_checkpoint(
                trainer_checkpoint_path=trainer_checkpoint_path,
                step=step,
            )
