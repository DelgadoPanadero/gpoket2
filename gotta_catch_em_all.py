import argparse
from pathlib import Path
from src.application.stg.generation import GenerationStep
from src.application.gym.train import train_pokemons
from src.application.gym.inference import PokemonGenerator
from src.application.slv.pokedex import PokedexStep
from src.application.brz.pokemon import PokemonStep
from src.application.gld.prof_oak_pc import ProfOakPcStep

from src.infra.stg.generation import VeekunAdapter
from src.infra.brz.pokemon import LocalPokemonRepository
from src.infra.slv.pokedex import LocalPokedexRepository
from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository
from src.infra.gym.checkpoints import LocalCheckpointStorageCallback
from src.infra.gym.checkpoints import LocalCheckpointStorageRepository


def main(args):
    result = []

    if args.stg:
        result = GenerationStep(
            adapter=VeekunAdapter(),
            generations=args.stg_generations,
        ).run()

    if args.brz:
        result = PokemonStep(
            pokemon_repository=LocalPokemonRepository(),
            adapter=VeekunAdapter(),
        ).run()

    if args.slv:
        result = PokedexStep(
            pokemon_repository=LocalPokemonRepository(),
            pokedex_repository=LocalPokedexRepository(),
        ).run()

    if args.gld:
        # partition = "box-" + datetime.now().strftime("%Y%m%d-%H%M"),
        result = ProfOakPcStep(
            pokedex_repository=LocalPokedexRepository(),
            profoakpc_repository=LocalProfOakPcRepository(
                partition=args.dataset_version,
            ),
        ).run()

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
        generator = PokemonGenerator(
            checkpoint_path=checkpoint_path,
            pokemon_repository=LocalPokemonRepository(base_dir=args.inference_output_dir),
            device=args.inference_device,
        )
        n = args.n_samples
        num_pokemon = generator.num_pokemon
        print(f"Generating {n} image(s) ({num_pokemon} Pokémon available)...")
        saved_paths = []
        for i in range(n):
            pokemon_idx = args.pokemon_idx if args.pokemon_idx is not None else i % num_pokemon
            temperature = args.inference_temperature + (i // num_pokemon) * 0.1
            saved_path, pokemon_idx = generator.generate(
                pokemon_idx=pokemon_idx,
                temperature=temperature,
                top_p=args.inference_top_p,
            )
            saved_paths.append(saved_path)
            if (i + 1) % 50 == 0 or n == 1:
                print(f"  [{i + 1}/{n}] Pokemon #{pokemon_idx} saved to {saved_path}")
        result = saved_paths

    return {"result": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokémon Local Operations")

    # --- Staging group ---
    stg_group = parser.add_argument_group("Staging layer")
    stg_group.add_argument(
        "--stg",
        action="store_true",
        help="Download Pokemon generation archives from veekun",
    )
    stg_group.add_argument(
        "--stg-generations",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Generations to download. Default: 1 2 3 4",
    )

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
        "--pixel-size",
        action="store_true",
        help="Pixel size of the sprites to be used. This is used to filter the dataset. Default: 64",
        default=64
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
        "--n-samples",
        type=int,
        default=1,
        help="Number of sprites to generate. Default: 1",
    )
    inference_group.add_argument(
        "--pokemon-idx",
        type=int,
        default=None,
        help="Pokémon index to generate. Cycles through all if not specified.",
    )
    inference_group.add_argument(
        "--inference-output-dir",
        type=str,
        default="data/gld/thinbaker_team",
        help="Output directory for generated sprites. Default: data/gld/thinbaker_team",
    )
    inference_group.add_argument(
        "--inference-temperature",
        type=float,
        default=1.2,
        help="Base sampling temperature. Each full cycle adds 0.1. Default: 1.2",
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
        default="cuda",
        help="Device for inference: cpu or cuda. Default: cuda",
    )

    args = parser.parse_args()

    print(main(args))
