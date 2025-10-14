import os
from pathlib import Path
from abc import ABC, abstractmethod
from src.domain.slv.pokedex import PokedexEntity


class PokedexRepository(ABC):
    @abstractmethod
    def load_one(
        self,
        pokedex_item_path: str,
    ) -> PokedexEntity:
        NotImplementedError()

    @abstractmethod
    def save_one(
        self,
        pokedex_item: PokedexEntity,
        pokedex_item_path: str,
    ) -> str:
        NotImplementedError()

    @abstractmethod
    def save_all(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> list[str]:
        NotImplementedError()

    @abstractmethod
    def load_all(
        self,
    ) -> list[PokedexEntity]:
        NotImplementedError()
