from pathlib import Path
from abc import ABC, abstractmethod
from src.domain.brz.pokemon import PokemonEntity


class PokemonRepository(ABC):
    @abstractmethod
    def load_one(
        self,
        img_path: str,
    ) -> PokemonEntity:
        NotImplementedError()

    @abstractmethod
    def load_all(
        self,
    ) -> list[PokemonEntity]:
        NotImplementedError()

    @abstractmethod
    def save_one(
        self,
        file_info: dict,
    ) -> str:
        NotImplementedError()

    @abstractmethod
    def save_all(
        self,
    ) -> list[str]:
        NotImplementedError()
