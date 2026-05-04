from abc import ABC, abstractmethod
from typing import Iterator

from src.domain.brz.pokemon import PokemonEntity


class GenerationAdapter(ABC):
    @abstractmethod
    def get_available_generations(self) -> list[int]:
        raise NotImplementedError()

    @abstractmethod
    def save_one(self, generation: int) -> str:
        raise NotImplementedError()

    @abstractmethod
    def save_all(self, generations: list[int]) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def extract_sprites(self) -> Iterator[PokemonEntity]:
        raise NotImplementedError()
