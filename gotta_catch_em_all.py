import argparse
from pathlib import Path
from src.application.train import train_pokemons
from src.application.slv.pokedex import get_pokedex
from src.application.brz.pokemon import get_pokemons
from src.application.gld.prof_oak_pc import get_prof_oak_pc
from src.application.inference import PokemonGenerator

from src.infra.brz.pokemon import LocalPokemonRepository
from src.infra.slv.pokedex import LocalPokedexRepository
from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository
from src.infra.train.checkpoints import LocalCheckpointStorageCallback
from src.infra.train.checkpoints import LocalCheckpointStorageRepository


def main(args):
    result = []

    if args.brz:
        result = get_pokemons(
            pokemon_repository=LocalPokemonRepository(),
        )

    if args.slv:
        result = get_pokedex(
            pokemon_repository=LocalPokemonRepository(),
            pokedex_repository=LocalPokedexRepository(),
        )

    if args.gld:
        # partition = "box-" + datetime.now().strftime("%Y%m%d-%H%M"),
        result = get_prof_oak_pc(
            pokedex_repository=LocalPokedexRepository(),
            profoakpc_repository=LocalProfOakPcRepository(
                partition=args.dataset_version,
            ),
        )

    if args.train:
        result = train_pokemons(
            profoakpc_repository=LocalProfOakPcRepository(
                partition=args.train_dataset_version,
            ),
            checkpoint_storage_callback=LocalCheckpointStorageCallback(
                dataset=args.train_dataset_version,
            ),
        )

    if args.inference:
        checkpoint_path = (
            Path(args.inference_checkpoint)
            if args.inference_checkpoint
            else LocalCheckpointStorageRepository().get_latest_checkpoint()
        )
        print(f"Loading checkpoint: {checkpoint_path}")
        generator = PokemonGenerator(checkpoint_path=checkpoint_path, device=args.inference_device)
        image, pokemon_idx = generator.generate(
            pokemon_idx=args.pokemon_idx,
            temperature=args.inference_temperature,
            top_p=args.inference_top_p,
        )
        output_path = Path(args.inference_output)
        image.save(output_path)
        print(f"Pokemon #{pokemon_idx} saved to {output_path}")
        result = str(output_path)

    return {"result": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokémon Local Operations")

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

    # --- Inference group ---
    inference_group = parser.add_argument_group("Inference layer")
    inference_group.add_argument(
        "--inference",
        action="store_true",
        help="Generate a Pokémon sprite",
    )
    inference_group.add_argument(
        "--inference-checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory for inference. Defaults to latest.",
    )
    inference_group.add_argument(
        "--pokemon-idx",
        type=int,
        default=None,
        help="Pokémon index to generate. Random if not specified.",
    )
    inference_group.add_argument(
        "--inference-output",
        type=str,
        default="sprite.png",
        help="Output PNG file path. Default: sprite.png",
    )
    inference_group.add_argument(
        "--inference-temperature",
        type=float,
        default=1.2,
        help="Sampling temperature. Default: 1.2",
    )
    inference_group.add_argument(
        "--inference-top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p. Default: 0.95",
    )
    inference_group.add_argument(
        "--inference-device",
        type=str,
        default="cpu",
        help="Device for inference: cpu or cuda. Default: cpu",
    )

    args = parser.parse_args()

    print(main(args))
