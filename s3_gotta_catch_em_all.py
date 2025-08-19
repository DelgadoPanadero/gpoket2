from src.application.train import train_pokemons
from src.application.slv.pokedex import get_pokedex
from src.application.brz.pokemon import get_pokemons
from src.application.gld.prof_oak_pc import get_prod_oak_pc

from src.infra.brz.pokemon import S3PokemonRepository
from src.infra.slv.pokedex import S3PokedexRepository
from src.infra.gld.prof_oak_pc import S3ProfOakPcRepository


if __name__=="__main__":

    #get_pokemons(
    #    S3PokemonRepository(),
    #)

    get_pokedex(
        S3PokemonRepository(),
        S3PokedexRepository(),
    )

    get_prod_oak_pc(
        S3PokedexRepository(), 
        S3ProfOakPcRepository(),
    )

    train_pokemons(
        S3ProfOakPcRepository(),
    )