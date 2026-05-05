import io
import re
import tarfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import requests

from src.domain.brz.pokemon import PokemonEntity
from src.application.stg.generation import GenerationAdapter


class VeekunAdapter(GenerationAdapter):
    _BASE_URL = "https://veekun.com/static/pokedex/downloads"

    def __init__(
        self,
        base_path: Path | str = Path("./data"),
    ):
        self.base_path = Path(base_path) / self.layer / self.entity_name
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_available_generations(self) -> list[int]:
        response = requests.get(self._BASE_URL)
        response.raise_for_status()
        generations = []
        for line in response.text.splitlines():
            match = re.search(r"generation-(\d+)", line)
            if match:
                generations.append(int(match.group(1)))
        return sorted(set(generations))

    def save_one(self, generation: int) -> str:
        response = requests.get(
            f"{self._BASE_URL}/generation-{generation}.tar.gz",
            stream=True,
        )
        response.raise_for_status()

        file_path = (
            self.base_path
            / f"gen{generation}"
            / f"generation-{generation}.tar.gz"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return str(file_path)

    def save_all(self, generations: list[int] = [1, 2, 3, 4]) -> list[str]:
        return [self.save_one(gen) for gen in generations]

    def _sprite_rel_path(self, gen: int, member_path: str) -> str | None:
        parts = Path(member_path).parts
        if (
            len(parts) < 4
            or parts[0] != "pokemon"
            or parts[1] != "main-sprites"
        ):
            return None

        game = parts[2]
        subdirs = parts[3:-1]
        stem = Path(parts[-1]).stem

        if "back" in subdirs:
            return None

        number_str, *variant_parts = stem.split("-", 1)
        try:
            number = int(number_str)
        except ValueError:
            return None

        variant = variant_parts[0].replace("-", "_") if variant_parts else ""
        name_parts = [f"{number:03d}"]
        if variant:
            name_parts.append(variant)
        name_parts.extend(subdirs)

        return f"gen{gen}/{game}/{'_'.join(name_parts)}.png"

    def extract_sprites(self) -> Iterator[PokemonEntity]:
        for tar_path in sorted(self.base_path.glob("**/generation-*.tar.gz")):
            match = re.search(r"generation-(\d+)", tar_path.name)
            if not match:
                continue
            gen = int(match.group(1))

            with tarfile.open(tar_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile() or not member.name.endswith(".png"):
                        continue
                    rel_path = self._sprite_rel_path(gen, member.name)
                    if rel_path is None:
                        continue

                    with tf.extractfile(member) as f:
                        data = f.read()

                    image = cv2.imdecode(
                        np.frombuffer(data, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if image is None:
                        continue

                    rel = Path(rel_path)
                    yield PokemonEntity(
                        name=rel.name,
                        generation=rel.parts[0],
                        game_name=rel.parts[1],
                        image=image,
                    )
