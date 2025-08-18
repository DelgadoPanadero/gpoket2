from src.application.train import train_pokemons
from src.application.slv.pokedex import get_pokedex
from src.application.gld.prof_oak_pc import get_prod_oak_pc

from src.infra.bzr.pokemon import LocalPokemonRepository
from src.infra.slv.pokedex import LocalPokedexRepository
from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository


if __name__=="__main__":

    get_pokedex(
        LocalPokemonRepository(),
        LocalPokedexRepository(),
    )

    get_prod_oak_pc(
        LocalPokedexRepository(),
        LocalProfOakPcRepository(),
    )

    train_pokemons(
        LocalProfOakPcRepository(),
    )