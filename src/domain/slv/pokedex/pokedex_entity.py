from pydantic import BaseModel, computed_field, field_validator


class PokedexEntity(BaseModel):
    name: str
    generation: str
    game_name: str
    data: str

    @computed_field
    @property
    def size(self) -> int:
        return len(self.data.split("\n"))

    @field_validator("data")
    @classmethod
    def validate_data(cls, v: str) -> str:
        rows = v.split("\n")
        width = len(rows[0].split(" "))
        for i, row in enumerate(rows):
            n = len(row.split(" "))
            if n != width:
                raise ValueError(f"row {i} has {n} tokens, expected {width}")
        return v
