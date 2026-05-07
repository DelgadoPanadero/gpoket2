from pydantic import BaseModel


class PokemonEntity(BaseModel):
    generation: str
    game_name: str
    name: str
    image: bytes
