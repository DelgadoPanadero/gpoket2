import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from pydantic import field_serializer


class PokemonEntity(BaseModel):
    name: str
    image: npt.NDArray[np.int8]

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("image")
    def serialize_image(self, value):
        return value.tolist()
