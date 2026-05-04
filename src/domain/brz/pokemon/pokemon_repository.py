from pathlib import Path
from abc import ABC, abstractmethod
from src.domain.brz.pokemon import PokemonEntity


class PokemonRepository(ABC):
    @abstractmethod
    def load_one(
        self,
        generation: str,
        game_name: str,
        name: str,
    ) -> PokemonEntity:
        NotImplementedError()

    @abstractmethod
    def load_all(
        self,
        generation: str | None = None,
        game_name: str | None = None,
    ) -> list[PokemonEntity]:
        NotImplementedError()

    @abstractmethod
    def save_one(
        self,
        pokemon: PokemonEntity,
    ) -> str:
        NotImplementedError()

    @abstractmethod
    def save_all(
        self,
        pokemons: list[PokemonEntity],
    ) -> list[str]:
        NotImplementedError()

    @abstractmethod
    def get_available_generations(self) -> list[str]:
        NotImplementedError()
