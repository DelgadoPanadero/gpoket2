import argparse
from pathlib import Path
from src.application.inference import PokemonGenerator


def get_latest_checkpoint(base_dir: Path) -> Path:
    checkpoints = sorted(
        [d for d in base_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    return checkpoints[-1]


def main(args):
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_latest_checkpoint(
        Path("/workspace/train/checkpoints/latest")
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    generator = PokemonGenerator(checkpoint_path=checkpoint_path, device=args.device)

    image, pokemon_idx = generator.generate(
        pokemon_idx=args.pokemon_idx,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_path = Path(args.output)
    image.save(output_path)
    print(f"Pokemon #{pokemon_idx} saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Pokémon sprite")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory. Defaults to latest.",
    )
    parser.add_argument(
        "--pokemon-idx",
        type=int,
        default=None,
        help="Pokémon index to generate. Random if not specified.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sprite.png",
        help="Output PNG file path. Default: sprite.png",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Sampling temperature. Default: 1.2",
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
        default="cpu",
        help="Device for inference: cpu or cuda. Default: cpu",
    )

    args = parser.parse_args()
    main(args)
