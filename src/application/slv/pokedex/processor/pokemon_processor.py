import cv2
import numpy as np
import numpy.typing as npt

from src.domain.brz.pokemon import PokemonEntity


class PokemonProcessor:
    def process(
        self,
        pokemon: PokemonEntity,
    ) -> PokemonEntity:
        if isinstance(pokemon.image, np.ndarray):
            image = pokemon.image
        else:
            image = cv2.imdecode(
                np.frombuffer(pokemon.image, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
        if image is None:
            raise ValueError(f"Could not decode image for {pokemon.name}")

        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3:4] / 255.0
            bgr = image[:, :, :3].astype(np.float32)
            white = np.full_like(bgr, 255, dtype=np.float32)
            image = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)

        return PokemonEntity(
            name=pokemon.name,
            generation=pokemon.generation,
            game_name=pokemon.game_name,
            image=cv2.imencode(".png", image)[1].tobytes(),
        )
