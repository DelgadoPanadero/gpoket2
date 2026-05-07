import os
from pathlib import Path
from src.domain.slv.pokedex import PokedexEntity
from src.domain.slv.pokedex import PokedexRepository


class LocalPokedexRepository(PokedexRepository):
    def __init__(
        self,
        base_path: Path | str = Path("./data"),
    ):
        self.base_path = (
            Path(base_path) /
            Path(self.layer) /
            Path(self.entity_name)
        )
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_one(
        self,
        pokedex_item: PokedexEntity,
    ) -> str:
        pokedex_item_path = (
            self.base_path
            / Path(pokedex_item.generation)
            / Path(pokedex_item.game_name)
            / Path(pokedex_item.name)
        )

        pokedex_item_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pokedex_item_path, "w") as file:
            file.write(pokedex_item.data)

        return pokedex_item_path

    def save_all(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> list[str]:

        pokedex_item_path_list = []
        for pokedex_item in pokedex_list:
            if pokedex_item_path := self.save_one(pokedex_item):
                pokedex_item_path_list.append(str(pokedex_item_path))

        return pokedex_item_path_list

    def load_one(
        self,
        generation: str,
        game_name: str,
        name: str,
    ) -> PokedexEntity:

        source_path = (
            self.base_path /
            Path(generation) /
            Path(game_name) /
            Path(name)
        )

        if not source_path.is_file() or source_path.suffix != ".txt":
            raise ValueError("Invalid pokedex item path")

        with open(source_path, "r") as file:
            data = file.read()

        return PokedexEntity(
            name=name,
            generation=generation,
            game_name=game_name,
            data=data,
        )

    def load_all(
        self,
        generation: str | None = None,
        game_name: str | None = None,
    ) -> list[PokedexEntity]:

        path_list = [path for path in self.base_path.glob("**/*.txt")]

        if generation is not None:
            path_list = [
                path
                for path in path_list
                if path.parent.parent.name == generation
            ]

        if game_name is not None:
            path_list = [
                path for path in path_list if path.parent.name == game_name
            ]

        if not path_list:
            raise ValueError("No valid pokedex item paths found")

        result_list = []
        for path in path_list:
            pokedex_item = self.load_one(
                generation=path.parent.parent.name,
                game_name=path.parent.name,
                name=path.name,
            )
            result_list.append(pokedex_item)

        return result_list
