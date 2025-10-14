from src.domain.brz.pokemon import PokemonRepository


def get_pokemons(
    pokemon_repository: PokemonRepository,
) -> list[str] | None:
    pokemon_path_list = pokemon_repository.save_all()

    return pokemon_path_list
