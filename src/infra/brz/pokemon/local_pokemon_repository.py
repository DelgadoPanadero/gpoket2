import cv2
import requests
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.domain.brz.pokemon import PokemonEntity
from src.domain.brz.pokemon import PokemonRepository


class LocalPokemonRepository(PokemonRepository):

    def __init__(
        self,
        base_dir: Path | str = Path("/home/data/brz/"),
        entity: str = "pokemon",
        partition: str = "",
    ):

        self.base_dir = Path(base_dir) / Path(entity) / Path(partition)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.api_url = (
            "https://api.github.com/repos/DelgadoPanadero/GPokeT2"
            "/contents/data/brz/pokemon?ref=main"
        )

    def load_one(
        self,
        img_path: str,
    ) -> PokemonEntity:

        image = cv2.imread(img_path)

        pokemon = PokemonEntity(
            name=Path(img_path).name,
            image=np.array(image),
        )

        return pokemon

    def load_all(
        self,
    ) -> list[PokemonEntity]:

        path_list = [path for path in self.base_dir.glob("**/*.png")]

        result_list = []
        for path in path_list:
            if result := self.load_one(str(path)):
                result_list.append(result)

        return result_list

    def save_one(
        self,
        file_info: dict,
    ) -> str:

        file_path = ""

        if file_info["type"] == "file" and file_info["name"].endswith(".png"):

            response = requests.get(file_info["download_url"])

            if response.status_code == 200:
                file_path = self.base_dir / file_info["name"]
                with open(file_path, "wb") as fin:
                    fin.write(response.content)

        return file_path

    def save_all(
        self,
    ) -> list[str]:

        response = requests.get(self.api_url)
        if response.status_code != 200:
            raise Exception(
                f"Error al acceder a {self.api_url}: {response.status_code}"
            )

        files = response.json()

        file_path_list = []
        for file_info in files:
            if file_path := self.save_one(file_info):
                file_path_list.append(file_path)

        return file_path_list
