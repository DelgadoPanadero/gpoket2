from src.application.train import train_pokemons
from src.application.slv.pokedex import get_pokedex
from src.application.brz.pokemon import get_pokemons
from src.application.gld.prof_oak_pc import get_prof_oak_pc

from src.infra.brz.pokemon import S3PokemonRepository
from src.infra.slv.pokedex import S3PokedexRepository
from src.infra.gld.prof_oak_pc import S3ProfOakPcRepository
from src.infra.train.checkpoints import S3CheckpointStorageCallback


if __name__=="__main__":

    get_pokemons(
        pokemon_repository=S3PokemonRepository(
            bucket = "brz",
            entity = "pokemons",
            partition = "",
        ),
    )

    get_pokedex(
        pokemon_repository=S3PokemonRepository(
            bucket = "brz",
            entity = "pokemons",
            partition = "",
        ),
        pokedex_repository=S3PokedexRepository(
            bucket = "slv",
            entity = "pokedex",
            partition = "",    
        ),
    )

    box_name = get_prof_oak_pc(
        pokedex_repository=S3PokedexRepository(
            bucket = "slv",
            entity = "pokedex",
            partition = "",    
        ),
        profoakpc_repository=S3ProfOakPcRepository(),
    )

    train_pokemons(
        profoakpc_repository = S3ProfOakPcRepository(),
        checkpoint_storage_callback = S3CheckpointStorageCallback(
        box_name = box_name,
        ),
    )