import numpy as np
from typing import Any
from pydantic import BaseModel, field_serializer


class PokemonEntity(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    generation: str
    game_name: str
    name: str
    image: Any  # np.ndarray (processing) or bytes (I/O)

    @field_serializer("image")
    def serialize_image(self, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @property
    def dim(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]
