from collections import Counter
from src.domain.slv.pokedex import PokedexEntity

EMPTY_THRESHOLD = 0.95


class EmptyValidator:
    def run(
        self,
        pokedex_entity: PokedexEntity,
    ) -> PokedexEntity | None:
        pixels = pokedex_entity.data.split()
        total = len(pixels)
        if total == 0:
            return None

        most_common_count = Counter(pixels).most_common(1)[0][1]
        if most_common_count / total >= EMPTY_THRESHOLD:
            return None

        return pokedex_entity
