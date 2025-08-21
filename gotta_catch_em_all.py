import argparse
from datetime import datetime
from src.application.train import train_pokemons
from src.application.slv.pokedex import get_pokedex
from src.application.brz.pokemon import get_pokemons
from src.application.gld.prof_oak_pc import get_prof_oak_pc

from src.infra.brz.pokemon import S3PokemonRepository
from src.infra.slv.pokedex import S3PokedexRepository
from src.infra.gld.prof_oak_pc import S3ProfOakPcRepository
from src.infra.train.checkpoints import S3CheckpointStorageCallback


def main(args):
    if args.brz:
        result = get_pokemons(
            pokemon_repository=S3PokemonRepository(
                bucket="brz",
                entity="pokemons",
                partition="",
            ),
        )

    if args.slv:
        result = get_pokedex(
            pokemon_repository=S3PokemonRepository(
                bucket="brz",
                entity="pokemons",
                partition="",
            ),
            pokedex_repository=S3PokedexRepository(
                bucket="slv",
                entity="pokedex",
                partition="",
            ),
        )

    if args.gld:
        #partition = "box-" + datetime.now().strftime("%Y%m%d-%H%M"),
        result = get_prof_oak_pc(
            pokedex_repository=S3PokedexRepository(
                bucket="slv",
                entity="pokedex",
                partition="",
            ),
            profoakpc_repository=S3ProfOakPcRepository(
                bucket="gld",
                prefix="prof_oak_pc",
                partition=args.dataset_version,
            ),
        )

    if args.train:
        train_pokemons(
            profoakpc_repository=S3ProfOakPcRepository(
                bucket="gld",
                prefix="prof_oak_pc",
                partition=args.train_dataset_version,
            ),
            checkpoint_storage_callback=S3CheckpointStorageCallback(
                box_name=args.train_dataset_version,
            ),
        )

    return {"result": result}


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Pokémon S3 Operations")

    # --- Bronze group ---
    bronze_group = parser.add_argument_group("Bronze layer")
    bronze_group.add_argument(
        "--brz",
        action="store_true",
        help="Get pokemons from Bronze",
    )

    # --- Silver group ---
    silver_group = parser.add_argument_group("Silver layer")
    silver_group.add_argument(
        "--slv",
        action="store_true",
        help="Get pokedex from Silver",
    )

    # --- Gold group ---
    gold_group = parser.add_argument_group("Gold layer")
    gold_group.add_argument(
        "--gld",
        action="store_true",
        help="Get Prof Oak PC from Gold",
    )
    gold_group.add_argument(
        "--dataset-version",
        type=str,
        help="Dataset version to use for training. Default `latest`",
        default="latest",
    )


    # --- Train group ---
    train_group = parser.add_argument_group("Train layer")
    train_group.add_argument(
        "--train",
        action="store_true",
        help="Train pokemons using Prof Oak PC",
    )
    train_group.add_argument(
        "--train-dataset-version",
        type=str,
        help="Dataset version to use for training. Default `latest`",
        default="latest",
    )

    args = parser.parse_args()

    print(main(args))