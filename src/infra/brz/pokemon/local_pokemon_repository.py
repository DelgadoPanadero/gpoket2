from pathlib import Path

import cv2
import numpy as np

from src.domain.brz.pokemon import PokemonEntity
from src.domain.brz.pokemon import PokemonRepository


class LocalPokemonRepository(PokemonRepository):
    def __init__(
        self,
        base_dir: Path | str = Path("/workspace/brz/"),
        entity: str = "pokemon",
    ):
        self.base_dir = Path(base_dir) / entity
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _is_valid_path(
        self,
        path: Path,
    ) -> bool:

        try:
            name = path.name
            generation = path.parent.parent.parent.name
            game_name = path.parent.parent.name
        except AttributeError:
            return False

        return path.is_file() and path.suffix == ".png"

    def get_available_generations(self) -> list[str]:
        generations = set()
        for path in self.base_dir.glob("**/*.png"):
            if self._is_valid_path(path):
                generations.add(path.parent.parent.parent.name)
        return sorted(generations)

    def load_one(
        self,
        generation: str,
        game_name: str,
        name: str,
    ) -> PokemonEntity:

        source_path = self.base_dir / generation / game_name / name
        if self._is_valid_path(source_path):
            image = cv2.imread(str(source_path))
        else:
            raise ValueError("Invalid image path")

        return PokemonEntity(
            name=source_path.name,
            image=np.array(image),
            generation=source_path.parent.parent.name,
            game_name=source_path.parent.name,
        )

    def load_all(
        self,
        generation: str | None = None,
        game_name: str | None = None,
    ) -> list[PokemonEntity]:

        paths = self.base_dir.glob("**/*.png")

        if generation is not None:
            paths = [
                path for path in paths if path.parent.parent.name == generation
            ]
        if game_name is not None:
            paths = [path for path in paths if path.parent.name == game_name]

        if not paths:
            raise ValueError("No valid image paths found")

        return [
            self.load_one(
                generation=path.parent.parent.name,
                game_name=path.parent.name,
                name=path.name,
            )
            for path in paths
        ]

    def save_one(
        self,
        pokemon: PokemonEntity,
    ) -> str:

        dest = (
            self.base_dir
            / pokemon.generation
            / pokemon.game_name
            / pokemon.name
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), pokemon.image)
        return str(dest)

    def save_all(self, pokemon_list: list[PokemonEntity]) -> list[str]:
        return [self.save_one(pokemon) for pokemon in pokemon_list]
