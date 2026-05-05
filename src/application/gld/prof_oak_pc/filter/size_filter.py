from src.domain.slv.pokedex import PokedexEntity


class SizeFilter:
    def __init__(
        self,
        size: int = 64,
    ):
        self.size = size

    def remove_blank_lines(
        self,
        data: str,
    ) -> str:

        data = [row.split() for row in data.split("\n")]

        # Remove blank lines
        data = [row for row in data if not all(item == "~" for item in row)]
        data = list(zip(*data))

        # Remove blank columns
        data = [col for col in data if not all(item == "~" for item in col)]
        data = list(zip(*data))

        return "\n".join(" ".join(cell for cell in row) for row in data)

    def pad_with_blanks(
        self,
        data: str,
    ) -> str:
        data = [row.split() for row in data.split("\n")]

        # Pad left columns with blank tokens
        col_to_pad = (self.size - len(data[0])) // 2
        data = [["~"] * col_to_pad + row + ["~"] * col_to_pad for row in data]

        # Pad right columns with blank tokens
        col_to_pad = self.size - len(data[0])
        data = [row + ["~"] * col_to_pad for row in data]

        # Pad top rows with blank lines
        row_to_pad = (self.size - len(data)) // 2
        data = [["~"] * self.size] * row_to_pad + data

        # Pad bottom rows with blank lines
        row_to_pad = self.size - len(data)
        data = data + [["~"] * self.size] * row_to_pad

        return "\n".join(" ".join(cell for cell in row) for row in data)

    def run(
        self,
        entity: PokedexEntity,
    ) -> PokedexEntity | None:

        data = entity.data

        data = self.remove_blank_lines(data)

        rows = data.split("\n")
        if len(rows) > self.size or len(rows[0].split()) > self.size:
            return None

        data = self.pad_with_blanks(data)

        return PokedexEntity(
            name=entity.name,
            generation=entity.generation,
            game_name=entity.game_name,
            data=data,
        )
