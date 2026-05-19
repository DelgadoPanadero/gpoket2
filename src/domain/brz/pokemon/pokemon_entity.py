import numpy as np
from typing import Any
from pydantic import BaseModel, field_serializer

from src.domain.brz.pokemon.pokemon_metadata import (
    EvolutionStage,
    PokemonType,
    Shininess,
)


class PokemonEntity(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    generation: str
    game_name: str
    name: str
    image: Any  # np.ndarray (processing) or bytes (I/O)
    type_1: PokemonType | None = None
    type_2: PokemonType | None = None
    shininess: Shininess = Shininess.NORMAL
    evolution_stage: EvolutionStage | None = None
    has_evolution: bool = False

    @field_serializer("image")
    def serialize_image(self, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @property
    def dim(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]
