import argparse
from pathlib import Path
from src.application.inference import PokemonGenerator
from src.infra.train.checkpoints import LocalCheckpointStorageRepository


def main(args):
    checkpoint_path = LocalCheckpointStorageRepository().get_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")

    generator = PokemonGenerator(checkpoint_path=checkpoint_path, device=args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n
    num_pokemon = generator.num_pokemon

    print(f"Generating {n} images ({num_pokemon} Pokémon available)...")

    for i in range(n):
        pokemon_idx = i % num_pokemon
        temperature = args.temperature + (i // num_pokemon) * 0.1

        image, _ = generator.generate(
            pokemon_idx=pokemon_idx,
            temperature=temperature,
            top_p=args.top_p,
        )

        output_path = output_dir / f"pokemon_{pokemon_idx:04d}_v{i // num_pokemon:02d}.png"
        image.save(output_path)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} generated")

    print(f"Done. Images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch generate Pokémon sprites from the latest checkpoint"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of images to generate. Default: 1000",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/gld/thinbaker_pc",
        help="Output directory. Default: data/gld/thinbaker_pc",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Base sampling temperature. Each full cycle adds 0.1. Default: 1.2",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p. Default: 0.95",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="Device for inference: cpu or cuda. Default: cpu",
    )

    args = parser.parse_args()
    main(args)
