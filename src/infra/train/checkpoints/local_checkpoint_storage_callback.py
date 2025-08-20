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
        output_dir: str,
    ) -> str | None:

        checkpoints = []
        for name in os.listdir(output_dir):
            match = re.match(r"checkpoint-(\d+)", name)
            if match:
                checkpoints.append((int(match.group(1)), name))

        last_checkpoint_path = None
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0])
            last_checkpoint_path = os.path.join(output_dir, checkpoints[-1][1])

        return last_checkpoint_path

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        os.makedirs(self.backup_dir, exist_ok=True)
        last_checkpoint_path = self._get_latest_checkpoint(self.backup_dir)
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

        checkpoint_dir = f"checkpoint-{state.global_step}"
        if state.is_world_process_zero and args.output_dir:
            last_checkpoint_path = os.path.join(args.output_dir, checkpoint_dir)
            shutil.move(last_checkpoint_path, self.backup_dir)
