from .pokemon_encoder import PokemonEncoder
from src.domain.brz.pokemon import PokemonRepository
from src.domain.slv.pokedex import PokedexRepository


def get_pokedex(
    pokemon_repository: PokemonRepository,
    pokedex_repository: PokedexRepository,
) -> None:

    pokemon_list = pokemon_repository.load_all()

    pokedex_list = []
    for pokemon_item in pokemon_list:
        pokedex_list.append(PokemonEncoder().run(pokemon_item))

    pokedex_repository.save_all(pokedex_list)


if __name__ == "__main__":

    from src.infra.bzr.pokemon import LocalPokemonRepository
    from src.infra.slv.pokedex import LocalPokedexRepository

    get_pokedex(
        pokemon_repository=LocalPokemonRepository(),
        pokedex_repository=LocalPokedexRepository(),
    )
