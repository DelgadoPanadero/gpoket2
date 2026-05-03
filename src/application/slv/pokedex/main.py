from src.application.shared import PokemonEncoder
from src.domain.brz.pokemon import PokemonRepository
from src.domain.slv.pokedex import PokedexRepository


def get_pokedex(
    pokemon_repository: PokemonRepository,
    pokedex_repository: PokedexRepository,
) -> list[str]:
    pokemon_list = pokemon_repository.load_all()

    encoder = PokemonEncoder()
    pokedex_list = []
    for pokemon_item in pokemon_list:
        if pokedex_item := encoder.run(pokemon_item):
            pokedex_list.append(pokedex_item)

    pokedex_item_path_list = pokedex_repository.save_all(pokedex_list)

    return pokedex_item_path_list
