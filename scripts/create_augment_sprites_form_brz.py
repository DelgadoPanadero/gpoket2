"""
Reads PNGs from data/brz/pokemon, applies the GLD augmentation pipeline
(SizeFilter + HorizontalFlip + ColorShift), and saves the resulting PNGs
to data/gld/augmentation preserving the gen/game_name folder structure.

Augmentation variants produced per original sprite:
  <name>.png          — original (padded to 64x64 grid)
  <name>_flip.png     — horizontal flip
  <name>_cyc1.png     — colour-cycle 1
  <name>_cyc1_flip.png
  <name>_cyc2.png     — colour-cycle 2
  <name>_cyc2_flip.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from src.domain.brz.pokemon import PokemonEntity
from src.domain.slv.pokedex import PokedexEntity
from src.application.slv.pokedex.processor import PokemonProcessor
from src.application.slv.pokedex.encoder import PokemonEncoder
from src.application.gld.prof_oak_pc.filter import SizeFilter
from src.application.gld.prof_oak_pc.augmentation import (
    HorizontalFlip,
    ColorShift,
)


BRZ_ROOT = PROJECT_ROOT / "data/brz/pokemon"
OUT_ROOT = PROJECT_ROOT / "data/gld/augmentation"


def load_png(path: Path) -> PokemonEntity:
    with open(path, "rb") as f:
        raw = f.read()
    parts = path.parts
    gen_idx = parts.index("pokemon") + 1
    return PokemonEntity(
        name=path.name,
        generation=parts[gen_idx],
        game_name=parts[gen_idx + 1],
        image=raw,
    )


def save_png(entity: PokedexEntity, out_dir: Path) -> None:
    encoder = PokemonEncoder()
    pokemon = encoder.decode(entity)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / pokemon.name
    img = cv2.imdecode(np.frombuffer(pokemon.image, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    processor = PokemonProcessor()
    encoder = PokemonEncoder()
    size_filter = SizeFilter(size=64)
    flip = HorizontalFlip()
    color_shift = ColorShift(
        permutations=ColorShift.ALL_PERMUTATIONS,
        suffixes=ColorShift.ALL_SUFFIXES,
    )

    png_paths = sorted(BRZ_ROOT.rglob("*.png"))
    if not png_paths:
        print(f"No PNGs found under {BRZ_ROOT}")
        return

    skipped = 0
    saved = 0

    for path in tqdm(png_paths, desc="Augmenting sprites"):
        pokemon = load_png(path)
        pokemon = processor.process(pokemon)
        entity = encoder.encode(pokemon)

        entity = size_filter.run(entity)
        if entity is None:
            skipped += 1
            continue

        out_dir = OUT_ROOT / entity.generation / entity.game_name

        variants: list[PokedexEntity] = [entity, flip.run(entity)]
        for shifted in color_shift.run(entity):
            variants.append(shifted)
            variants.append(flip.run(shifted))

        for variant in variants:
            save_png(variant, out_dir)
            saved += 1

    print(
        f"Done — {saved} images saved to {OUT_ROOT}, {skipped} sprites skipped (too large).",
    )


if __name__ == "__main__":
    main()
