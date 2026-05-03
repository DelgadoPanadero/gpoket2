from pydantic import BaseModel, field_validator


class PokedexEntity(BaseModel):
    name: str
    data: str

    @field_validator("data")
    @classmethod
    def validate_data(cls, v: str) -> str:
        """
        Validates the encoded sprite format: 64 rows × 64 tokens = 4096 tokens
        total. Each row starts with a two-digit row number ("00"–"63") at
        position i*64, followed by 63 pixel characters.
        """
        tokens = v.split()

        if len(tokens) != 4096:
            raise ValueError(f"data must have 4096 tokens, got {len(tokens)}")

        for i in range(64):
            real_row_number = "%02d" % i
            pred_row_number = tokens[i * 64]
            if pred_row_number != real_row_number:
                raise ValueError(
                    f"token at position {i * 64} must be '{real_row_number}'"
                    f", got '{pred_row_number}'"
                )

        return v
