import os
from pathlib import Path
from abc import ABC, abstractmethod
from src.domain.slv.pokedex import PokedexEntity


class PokedexRepository(ABC):
    @abstractmethod
    def load_one(
        self,
        generation: str,
        game_name: str,
        name: str,
    ) -> PokedexEntity:
        NotImplementedError()

    @abstractmethod
    def save_one(
        self,
        pokedex_item: PokedexEntity,
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
        generation: str | None = None,
        game_name: str | None = None,
    ) -> list[PokedexEntity]:
        NotImplementedError()
