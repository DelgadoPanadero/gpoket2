from src.application.train import train_pokemons
from src.application.slv.pokedex import get_pokedex
from src.application.brz.pokemon import get_pokemons
from src.application.gld.prof_oak_pc import get_prof_oak_pc

from src.infra.brz.pokemon import LocalPokemonRepository
from src.infra.slv.pokedex import LocalPokedexRepository
from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository
from src.infra.train.checkpoints import LocalCheckpointStorageCallback


if __name__ == "__main__":

    pokemon_item_paths = get_pokemons(
        LocalPokemonRepository(
            base_dir="/home/data/brz/",
            entity="pokemon",
            partition="",
        ),
    )

    pokedex_item_paths = get_pokedex(
        LocalPokemonRepository(
            base_dir="/home/data/brz/",
            entity="pokemon",
            partition="",
        ),
        LocalPokedexRepository(
            base_dir="/home/data/slv",
            entity="pokedex",
            partition="",
        ),
    )

    box_name = get_prof_oak_pc(
        LocalPokedexRepository(
            base_dir="/home/data/slv",
            entity="pokedex",
            partition="",
        ),
        LocalProfOakPcRepository(
            base_dir="/home/data/gld", entity="prof_oak_pc"
        ),
    )

    train_pokemons(
        LocalProfOakPcRepository(
            base_dir="/home/data/gld",
            entity="prof_oak_pc",
            partition="",
        ),
        LocalCheckpointStorageCallback(
            base_dir="/home/data/train",
            dataset="",
        ),
    )
