from collections import Counter
from src.domain.slv.pokedex import PokedexEntity


class ColorValidator:
    def run(
        self,
        pokedex_entity: PokedexEntity,
    ) -> PokedexEntity | None:
        rows = [row.split() for row in pokedex_entity.data.split("\n")]

        corners = [
            rows[0][0],
            rows[0][-1],
            rows[-1][0],
            rows[-1][-1],
        ]

        if not all(c == "~" for c in corners):
            return None

        margins = (
            rows[0][1:-1]
            + [row[0] for row in rows[1:-1]]
            + [row[-1] for row in rows[1:-1]]
            + rows[-1][1:-1]
        )

        if Counter(margins).most_common(1)[0][0] != "~":
            return None

        return pokedex_entity
