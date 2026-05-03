from pydantic import BaseModel, field_validator


class PokedexEntity(BaseModel):
    name: str
    data: str

    @field_validator("data")
    @classmethod
    def validate_data(cls, v: str) -> str:
        """
        Validates the encoded sprite format: 64 rows × 64 pixel characters.
        Each row contains 64 space-separated tokens (no row numbers at this
        stage — those are injected later by the Gold tokenizer).
        """
        rows = v.splitlines()

        if len(rows) != 64:
            raise ValueError(f"data must have 64 rows, got {len(rows)}")

        for i, row in enumerate(rows):
            tokens = row.split()
            if len(tokens) != 64:
                raise ValueError(
                    f"row {i} must have 64 tokens, got {len(tokens)}"
                )

        return v
