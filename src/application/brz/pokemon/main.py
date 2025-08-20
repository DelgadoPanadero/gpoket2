from src.domain.brz.pokemon import PokemonRepository


def get_pokemons(
    pokemon_repository: PokemonRepository,
) -> list[str] | None:

    pokemon_path_list = pokemon_repository.save_all()

    return pokemon_path_list


if __name__ == "__main__":

    from src.infra.brz.pokemon import LocalPokemonRepository

    get_pokemons(
        pokemon_repository=LocalPokemonRepository(),
    )
