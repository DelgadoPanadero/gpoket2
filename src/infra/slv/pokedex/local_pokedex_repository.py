import os
from pathlib import Path
from src.domain.slv.pokedex import PokedexEntity
from src.domain.slv.pokedex import PokedexRepository


class LocalPokedexRepository(PokedexRepository):

    def __init__(
        self,
        base_dir: Path | str = Path("/home/data/slv"),
        entity: str = "pokedex",
        partition: str = "",
    ):
        self.base_dir = Path(base_dir) / Path(entity) / Path(partition)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_one(
        self,
        pokedex_item_path: str,
    ) -> PokedexEntity:

        with open(pokedex_item_path, "r") as file:
            data = file.read()

        return PokedexEntity(
            name=Path(pokedex_item_path).name,
            data=data,
        )

    def save_one(
        self,
        pokedex_item: PokedexEntity,
        pokedex_item_path: str,
    ) -> str:

        with open(pokedex_item_path, "w") as file:
            file.write(pokedex_item.data)

        return pokedex_item_path

    def save_all(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> list[str]:

        pokedex_item_path_list = []
        for pokedex_item in pokedex_list:

            pokedex_item_path = str(self.base_dir / Path(pokedex_item.name))

            if pokedex_item_path := self.save_one(
                pokedex_item=pokedex_item,
                pokedex_item_path=pokedex_item_path,
            ):

                pokedex_item_path_list.append(pokedex_item_path)

        return pokedex_item_path_list

    def load_all(
        self,
    ) -> list[PokedexEntity]:

        path_list = [path for path in self.base_dir.glob("**/*.txt")]

        result_list = []
        for path in path_list:
            if result := self.load_one(str(path)):
                result_list.append(result)

        return result_list
