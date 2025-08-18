from pydantic import BaseModel


class PokedexEntity(BaseModel):
    name: str
    data: str
