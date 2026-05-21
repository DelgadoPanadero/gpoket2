from src.domain.slv.pokedex import PokedexEntity


class ColorShift:
    # Each tuple is (r_src, g_src, b_src): indices into the original (r,g,b)
    # that fill the new (r,g,b). The identity (0,1,2) is omitted.
    ALL_PERMUTATIONS: list[tuple[int, int, int]] = [
        (1, 0, 2),  # swap R↔G
        (2, 1, 0),  # swap R↔B
        (0, 2, 1),  # swap G↔B
        (1, 2, 0),  # cycle R→G→B→R
        (2, 0, 1),  # cycle R→B→G→R
    ]
    ALL_SUFFIXES: list[str] = ["_rg", "_rb", "_gb", "_cyc1", "_cyc2"]

    PERMUTATIONS = ALL_PERMUTATIONS
    SUFFIXES = ALL_SUFFIXES

    def __init__(
        self,
        permutations: list[tuple[int, int, int]] | None = None,
        suffixes: list[str] | None = None,
    ):
        self._permutations = (
            permutations if permutations is not None else self.PERMUTATIONS
        )
        self._suffixes = suffixes if suffixes is not None else self.SUFFIXES
        self._tables = [self._build_table(perm) for perm in self._permutations]

    @staticmethod
    def _build_table(perm: tuple[int, int, int]) -> dict[int, int]:
        table: dict[int, int] = {}
        for idx in range(64):
            r, g, b = idx // 16, (idx % 16) // 4, idx % 4
            channels = (r, g, b)
            nr, ng, nb = channels[perm[0]], channels[perm[1]], channels[perm[2]]
            table[idx + 59] = nr * 16 + ng * 4 + nb + 59
        return table

    def run(self, entity: PokedexEntity) -> list[PokedexEntity]:
        base = entity.name.replace(".txt", "")
        return [
            PokedexEntity(
                name=f"{base}{suffix}.txt",
                generation=entity.generation,
                game_name=entity.game_name,
                data=entity.data.translate(table),
            )
            for table, suffix in zip(self._tables, self._suffixes)
        ]
