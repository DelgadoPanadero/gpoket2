import os
import cv2
import numpy as np
from pathlib import Path

from src.domain.brz.pokemon import PokemonEntity
from src.domain.brz.pokemon import PokemonRepository


class LocalPokemonRepository(PokemonRepository):

    def load_one(
        self,
        img_path: Path,
    ) -> PokemonEntity:

        image = cv2.imread(str(img_path))

        pokemon = PokemonEntity(
            name=img_path.name,
            image=np.array(image),
        )

        return pokemon

    def load_all(
        self,
    ) -> list[PokemonEntity]:

        source_dir = "/home/data/bzr/pokemons/"
        os.makedirs(source_dir, exist_ok=True)
        paths = [x for x in Path(source_dir).glob("**/*.png")]

        result = []
        for path in paths:
            result.append(self.load_one(path))

        return result
