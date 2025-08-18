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

        width, height, _ = image.shape

        array = []
        for y in range(0, height):

            row = []
            # row = ["%02d" % (y)]
            for x in range(0, width):
                r, g, b = image[y, x] // 64
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
