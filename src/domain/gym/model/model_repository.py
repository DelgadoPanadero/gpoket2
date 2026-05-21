from pathlib import Path
from abc import ABC, abstractmethod

from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.gym.model.model_card import ModelCard


class ModelRepository(ABC):
    layer = "gym"
    entity_name = "model"

    @abstractmethod
    def upload(
        self,
        checkpoint_path: Path | str,
        repo_id: str,
        version: str | None = None,
        model_card: ModelCard | None = None,
        model_code_path: Path | str | None = None,
    ) -> str:
        NotImplementedError()

    @abstractmethod
    def upload_dataset(
        self,
        box_entity: BoxEntity,
        repo_id: str,
    ) -> str:
        NotImplementedError()
