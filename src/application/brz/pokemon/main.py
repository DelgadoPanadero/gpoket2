from src.domain.brz.pokemon import PokemonRepository


def get_pokemons(
    pokemon_repository: PokemonRepository,
) -> None:

    pokemon_repository.save_all()


if __name__ == "__main__":

    from src.infra.brz.pokemon import LocalPokemonRepository

    get_pokemons(
        pokemon_repository=LocalPokemonRepository(),
    )
