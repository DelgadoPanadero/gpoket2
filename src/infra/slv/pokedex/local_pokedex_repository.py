import os
from pathlib import Path
from src.domain.slv.pokedex import PokedexEntity
from src.domain.slv.pokedex import PokedexRepository


class LocalPokedexRepository(PokedexRepository):

    source_dir = "/home/data/slv/pokedex"

    def load_one(
        self,
        img_path: Path,
    ) -> PokedexEntity:

        with open(img_path, "r") as file:
            data = file.read()

        return PokedexEntity(name=img_path.name, data=data)

    def save_one(
        self,
        pokedex_item: PokedexEntity,
    ) -> None:

        os.makedirs(self.source_dir, exist_ok=True)
        with open(f"{self.source_dir}/{pokedex_item.name}", "w") as file:
            file.write(pokedex_item.data)

    def save_all(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> None:

        for pokedex_item in pokedex_list:
            self.save_one(pokedex_item)

    def load_all(
        self,
    ) -> list[PokedexEntity]:

        os.makedirs(self.source_dir, exist_ok=True)
        paths = [x for x in Path(self.source_dir).glob("**/*.txt")]

        result = []
        for path in paths:
            result.append(self.load_one(path))

        return result
