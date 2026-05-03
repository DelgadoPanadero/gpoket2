import numpy as np
import numpy.typing as npt

from src.domain.slv.pokedex import PokedexEntity
from src.domain.brz.pokemon import PokemonEntity


class PokemonEncoder:
    def __init__(
        self,
        n_characters: int = 64,
    ):
        self.n_characters = n_characters

    @staticmethod
    def _encode(
        image: npt.NDArray[np.int8],
    ) -> list[list[str]]:
        """ """

        height, width, _ = image.shape

        array = []
        for y in range(0, height):
            row = []
            # row = ["%02d" % (y)]
            for x in range(0, width):
                b, g, r = image[y, x] // 64
                is_blank = min(image[y, x]) > 245 or max(image[y, x]) < 10
                char = (
                    "~"
                    if is_blank
                    else chr(r * 4**2 + g * 4**1 + b * 4**0 + 59)
                )

                row.append(char)
            array.append(row)
            # array.append(row[:-1])

        return array

    @staticmethod
    def _decode(
        array: list[list[str]],
    ) -> npt.NDArray[np.uint8]:
        height = len(array)
        width = len(array[0]) if array else 0
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        for y, row in enumerate(array):
            for x, char in enumerate(row):
                if char != "~":
                    idx = ord(char) - 59
                    r = idx // 16
                    g = (idx % 16) // 4
                    b = idx % 4
                    img[y, x] = [b * 85, g * 85, r * 85]  # BGR
        return img

    @staticmethod
    def _array_to_text(
        array: list[list[str]],
    ) -> str:
        return "\n".join([" ".join(r) for i, r in enumerate(array)])

    def run(
        self,
        pokemon: PokemonEntity,
    ) -> PokedexEntity:
        image = pokemon.image

        pokedex_data_array = self._encode(image)

        pokedex_data_text = self._array_to_text(pokedex_data_array)

        return PokedexEntity(
            name=pokemon.name.replace(".png", ".txt"),
            data=pokedex_data_text,
        )
