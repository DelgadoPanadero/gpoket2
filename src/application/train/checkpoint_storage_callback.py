from abc import ABC
from transformers import TrainerCallback  # type: ignore


class CheckpointStorageCallback(ABC, TrainerCallback):

    resume_from_checkpoint: str | None = None
