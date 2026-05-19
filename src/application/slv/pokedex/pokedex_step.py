from src.domain.brz.pokemon import PokemonRepository
from src.domain.slv.pokedex import PokedexRepository
from src.application.slv.pokedex.encoder import PokemonEncoder
from src.application.slv.pokedex.processor import PokemonProcessor
from src.application.slv.pokedex.validator import ColorValidator
from src.application.slv.pokedex.validator import EmptyValidator


class PokedexStep:
    def __init__(
        self,
        pokemon_repository: PokemonRepository,
        pokedex_repository: PokedexRepository,
        generations: list[str] | None = None,
    ):
        self.pokemon_repository = pokemon_repository
        self.pokedex_repository = pokedex_repository
        self.available_generations = ["gen1", "gen2", "gen3", "gen4"]

    def run(
        self,
        generations: list[str] = ["gen3", "gen4"],
    ) -> list[str]:

        if any(
            generation not in self.available_generations
            for generation in generations
        ):
            raise ValueError(
                f"Invalid generation specified. Supported generations are "
                f"{self.available_generations}.",
            )

        pokemon_list = []
        for generation in generations:
            pokemon_list += self.pokemon_repository.load_all(
                generation=generation,
            )

        processor = PokemonProcessor()
        for i, pokemon in enumerate(pokemon_list):
            pokemon_list[i] = processor.process(pokemon)

        encoder = PokemonEncoder()
        pokedex_list = []
        for pokemon_item in pokemon_list:
            if pokedex_item := encoder.encode(pokemon_item):
                pokedex_list.append(pokedex_item)

        color_validator = ColorValidator()
        empty_validator = EmptyValidator()
        pokedex_list = [
            pokedex_item
            for pokedex_item in pokedex_list
            if (
                color_validator.run(pokedex_item)
                and empty_validator.run(pokedex_item)
            )
        ]

        return self.pokedex_repository.save_all(pokedex_list)
