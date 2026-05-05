from abc import ABC
import os
import re
import shutil
from pathlib import Path

from transformers import TrainerState  # type: ignore
from transformers import TrainerControl  # type: ignore
from transformers import TrainerCallback  # type: ignore
from transformers import TrainingArguments  # type: ignore
from transformers import TrainerCallback  # type: ignore


class CheckpointStorageCallback(ABC, TrainerCallback):
    resume_from_checkpoint: str | None = None

    def __init__(
        self,
        checkpoint_storage_adapter,
    ):
        self.resume_from_checkpoint = None
        self.checkpoint_storage_adapter = checkpoint_storage_adapter

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        last_checkpoint_path = (
            self.checkpoint_storage_adapter.get_latest_checkpoint()
        )
        if last_checkpoint_path and args.output_dir:
            self.checkpoint_storage_adapter.load_checkpoint(
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
            self.checkpoint_storage_adapter.save_checkpoint(
                os.path.join(
                    args.output_dir,
                    f"checkpoint-{state.global_step}",
                ),
            )
