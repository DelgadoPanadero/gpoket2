from pathlib import Path
from abc import ABC, abstractmethod
from src.domain.brz.pokemon import PokemonEntity


class PokemonRepository(ABC):

    @abstractmethod
    def load_one(
        self,
        img_path: Path,
    ) -> PokemonEntity:

        NotImplementedError()

    @abstractmethod
    def load_all(
        self,
    ) -> list[PokemonEntity]:

        NotImplementedError()
