from src.domain.brz.pokemon import PokemonRepository
from src.application.stg.generation import GenerationAdapter


class PokemonStep:
    def __init__(
        self,
        pokemon_repository: PokemonRepository,
        pokemon_extraction_adapter: GenerationAdapter,
    ):
        self.pokemon_repository = pokemon_repository
        self.adapter = pokemon_extraction_adapter

    def run(self) -> list[str]:
        result = []
        for entity in self.adapter.extract_sprites():
            target_path = self.pokemon_repository.save_one(entity)
            result.append(target_path)
        return result
