import argparse
from pathlib import Path
from src.application.stg.generation import GenerationStep
from src.application.gym.train import PokemonTrainerStep
from src.application.gym.inference import PokemonGenerator
from src.application.slv.pokedex import PokedexStep
from src.application.brz.pokemon import PokemonStep
from src.application.gld.prof_oak_pc import ProfOakPcStep

from src.infra.stg.generation import VeekunAdapter
from src.infra.brz.pokemon import LocalPokemonRepository
from src.infra.slv.pokedex import LocalPokedexRepository
from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository
from src.infra.gym.checkpoints import LocalCheckpointStorageAdapter


def main(args):
    result = []

    base_path = "/workspace"

    if args.stg:
        result = GenerationStep(
            generations=args.stg_generations,
            adapter=VeekunAdapter(base_path=base_path),
        ).run()

    if args.brz:
        result = PokemonStep(
            pokemon_repository=LocalPokemonRepository(base_path=base_path),
            pokemon_extraction_adapter=VeekunAdapter(base_path=base_path),
        ).run()

    if args.slv:
        result = PokedexStep(
            pokemon_repository=LocalPokemonRepository(base_path=base_path),
            pokedex_repository=LocalPokedexRepository(base_path=base_path),
        ).run()

    if args.gld:
        # partition = "box-" + datetime.now().strftime("%Y%m%d-%H%M"),
        result = ProfOakPcStep(
            context_length=args.context_length,
            pokedex_repository=LocalPokedexRepository(
                base_path=base_path,
            ),
            profoakpc_repository=LocalProfOakPcRepository(
                base_path=base_path,
                partition=args.dataset_version,
            ),
        ).run()

    if args.train:
        result = PokemonTrainerStep(
            profoakpc_repository=LocalProfOakPcRepository(
                partition=args.train_dataset_version,
                base_path=base_path,
            ),
            checkpoint_storage_adapter=LocalCheckpointStorageAdapter(
                dataset=args.train_dataset_version,
                base_path=args.checkpoint_base_path,
            ),
            context_length=args.context_length,
            output_dir=str(Path(args.checkpoint_base_path) / "trainer_tmp"),
        ).run()

    if args.inference:
        checkpoint_path = (
            Path(args.inference_checkpoint)
            if args.inference_checkpoint
            else LocalCheckpointStorageAdapter(
                base_path=args.checkpoint_base_path
            ).get_latest_checkpoint()
        )
        print(f"Loading checkpoint: {checkpoint_path}")
        generator = PokemonGenerator(
            checkpoint_path=checkpoint_path,
            device=args.inference_device,
            pokemon_repository=LocalPokemonRepository(
                base_path=args.inference_output_dir,
            ),
        )
        n = args.n_samples
        print(f"Generating {n} image(s)...")
        saved_paths = []
        pokemon_idx = int(args.name) if args.name is not None else None
        for i in range(n):
            saved_path, cond_meta = generator.generate(
                pokemon_idx=pokemon_idx,
                type1=args.type1,
                type2=args.type2,
                is_shiny=args.is_shiny,
                generation=args.generation,
                evolution_stage=args.evolution_stage,
                has_evolution=args.has_evolution,
                novel=pokemon_idx is None,
                temperature=args.inference_temperature,
                top_p=args.inference_top_p,
            )
            saved_paths.append(saved_path)
            if (i + 1) % 50 == 0 or n == 1:
                print(f"  [{i + 1}/{n}] {cond_meta} saved to {saved_path}")
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
        default=64,
    )
    gold_group.add_argument(
        "--dataset-version",
        type=str,
        help="Dataset version to use for training. Default `latest`",
        default="latest",
    )
    gold_group.add_argument(
        "--context-length",
        type=int,
        help="Context length (tokens per chunk) for tokenization. Default: 4096",
        default=4096,
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
    train_group.add_argument(
        "--checkpoint-base-path",
        type=str,
        default="/workspace",
        help="Base path for checkpoint storage. Default: /workspace",
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
        "--name",
        type=str,
        default=None,
        help="Pokemon name to reproduce (e.g. '025'). Random if not specified.",
    )
    inference_group.add_argument(
        "--type1",
        type=int,
        default=None,
        help="Primary type index (0-17). Random if not specified.",
    )
    inference_group.add_argument(
        "--type2",
        type=int,
        default=None,
        help="Secondary type index (0-17, 18=none). Random if not specified.",
    )
    inference_group.add_argument(
        "--is-shiny",
        type=int,
        default=None,
        choices=[0, 1],
        help="Shiny flag (0=normal, 1=shiny). Random if not specified.",
    )
    inference_group.add_argument(
        "--generation",
        type=int,
        default=None,
        help="Generation index 0-based (gen3=2, gen4=3). Random if not specified.",
    )
    inference_group.add_argument(
        "--evolution-stage",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Evolution stage 0-based (0=base, 1=stage2, 2=stage3). Random if not specified.",
    )
    inference_group.add_argument(
        "--has-evolution",
        type=int,
        default=None,
        choices=[0, 1],
        help="Has evolution (0=no, 1=yes). Random if not specified.",
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
