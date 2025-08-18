import os
from pathlib import Path
from abc import ABC, abstractmethod
from src.domain.slv.pokedex import PokedexEntity


class PokedexRepository(ABC):

    source_dir = ""

    @abstractmethod
    def load_one(
        self,
        img_path: Path,
    ) -> PokedexEntity:

        NotImplementedError()

    @abstractmethod
    def save_one(
        self,
        pokedex_item: PokedexEntity,
    ) -> None:

        NotImplementedError()

    @abstractmethod
    def save_all(self, pokedex_list: list[PokedexEntity]) -> None:

        NotImplementedError()

    @abstractmethod
    def load_all(
        self,
    ) -> list[PokedexEntity]:

        NotImplementedError()
