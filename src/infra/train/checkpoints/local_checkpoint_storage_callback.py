import os
import re
import shutil
from transformers import TrainerState  # type: ignore
from transformers import TrainerControl  # type: ignore
from transformers import TrainerCallback  # type: ignore
from transformers import TrainingArguments  # type: ignore


class LocalCheckpoitStorageCallback(TrainerCallback):

    def __init__(
        self,
        dataset_name: str,
    ):
        self.backup_dir = f"/home/data/train/checkpoints/{dataset_name}"

    def _get_latest_checkpoint(
        self,
    ) -> str | None:

        checkpoints = []
        for file_name in os.listdir(self.backup_dir):
            if match := re.match(r"checkpoint-(\d+)", file_name):
                checkpoints.append(
                    {
                        "step" : int(match.group(1)),
                        "checkpoint_path": os.path.join(
                            self.backup_dir,
                            file_name,
                        )
                    },
                )

        last_checkpoint_path = None
        if checkpoints:
            checkpoints.sort(key=lambda x: x["step"])
            last_checkpoint_path = checkpoints[-1]["checkpoint_path"]

        return last_checkpoint_path

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        os.makedirs(self.backup_dir, exist_ok=True)
        last_checkpoint_path = self._get_latest_checkpoint()
        if last_checkpoint_path and args.output_dir:
            shutil.move(last_checkpoint_path, args.output_dir)
            args.resume_from_checkpoint = last_checkpoint_path

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):

        checkpoint_name = f"checkpoint-{state.global_step}"
        if state.is_world_process_zero and args.output_dir:
            last_checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
            shutil.move(last_checkpoint_path, self.backup_dir)
