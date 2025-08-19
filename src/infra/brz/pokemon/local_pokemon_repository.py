import os
import cv2
import numpy as np
import requests
from pathlib import Path

from src.domain.brz.pokemon import PokemonEntity
from src.domain.brz.pokemon import PokemonRepository


class LocalPokemonRepository(PokemonRepository):

    def __init__(self):

        self.source_dir = Path("/home/data/brz/pokemon/")
        self.source_dir.mkdir(parents=True, exist_ok=True)

        # Datos del origen
        self.owner = "DelgadoPanadero"
        self.repo = "GPokeT2"
        self.branch = "main"
        self.folder = "data/bzr/pokemons"
        self.api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{self.folder}?ref={self.branch}"

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

        paths = [x for x in self.source_dir.glob("**/*.png")]

        result = []
        for path in paths:
            result.append(self.load_one(path))

        return result

    def save_one(
        self,
        file_info: dict,
    ) -> str:

        file_path = ""

        if file_info["type"] == "file" and file_info["name"].endswith(".png"):

            response = requests.get(file_info["download_url"])

            if response.status_code == 200:
                file_path = self.source_dir / file_info["name"]
                with open(file_path, "wb") as f:
                    f.write(response.content)

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
            file_path = self.save_one(file_info)
            file_path_list += [file_path] if file_path else []

        return file_path_list
