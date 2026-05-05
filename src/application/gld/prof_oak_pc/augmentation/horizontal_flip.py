from src.domain.slv.pokedex import PokedexEntity


class HorizontalFlip:
    def run(self, entity: PokedexEntity) -> PokedexEntity:
        rows = entity.data.splitlines()
        flipped_rows = [" ".join(row.split()[::-1]) for row in rows]
        return PokedexEntity(
            name=entity.name.replace(".txt", "_flip.txt"),
            generation=entity.generation,
            game_name=entity.game_name,
            data="\n".join(flipped_rows),
        )
