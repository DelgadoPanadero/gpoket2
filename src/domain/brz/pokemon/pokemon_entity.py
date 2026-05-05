import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from pydantic import computed_field
from pydantic import field_serializer


class PokemonEntity(BaseModel):
    generation: str
    game_name: str
    name: str
    image: npt.NDArray[np.int8]

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def dim(self) -> tuple[int, int] | None:
        if self.image.ndim >= 2:
            return (self.image.shape[0], self.image.shape[1])
        return None

    @field_serializer("image")
    def serialize_image(self, value):
        return value.tolist()
