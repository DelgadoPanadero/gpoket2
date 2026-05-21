import re
import json
from pathlib import Path

from src.domain.brz.pokemon.pokemon_metadata import (
    NUM_GENERATIONS,
    PokemonColor,
    PokemonHabitat,
    PokemonShape,
    PokemonType,
    EvolutionStage,
    Shininess,
)

# ── Types ──────────────────────────────────────────────────────────────────────
TYPE_TO_IDX: dict[str, int] = {t.name.lower(): t.value for t in PokemonType}
TYPE_TO_POKEMON_TYPE: dict[str, PokemonType] = {
    t.name.lower(): t for t in PokemonType
}
TYPE1_UNK = len(PokemonType)  # 18
TYPE2_NONE = len(PokemonType)  # 18 — pokémon sin tipo secundario
TYPE2_UNK = len(PokemonType) + 1  # 19
NUM_TYPES1 = len(PokemonType) + 1  # 19 (0-17 tipos + UNK)
NUM_TYPES2 = len(PokemonType) + 2  # 20 (0-17 tipos + NONE + UNK)

# ── Colors ─────────────────────────────────────────────────────────────────────
COLOR_TO_IDX: dict[str, int] = {c.name.lower(): c.value for c in PokemonColor}
COLOR_UNK = len(PokemonColor)
NUM_COLORS = len(PokemonColor) + 1

# ── Shapes ─────────────────────────────────────────────────────────────────────
SHAPE_TO_IDX: dict[str, int] = {
    s.name.lower().replace("_", "-"): s.value for s in PokemonShape
}
SHAPE_UNK = len(PokemonShape)
NUM_SHAPES = len(PokemonShape) + 1

# ── Habitats ───────────────────────────────────────────────────────────────────
HABITAT_TO_IDX: dict[str, int] = {
    h.name.lower().replace("_", "-"): h.value for h in PokemonHabitat
}
HABITAT_NONE = len(PokemonHabitat)
HABITAT_UNK = len(PokemonHabitat) + 1
NUM_HABITATS = len(PokemonHabitat) + 2

# ── Other counts ───────────────────────────────────────────────────────────────
NUM_EVO_STAGES = len(EvolutionStage) + 1  # stages + UNK
NUM_HAS_EVOLUTION = len(Shininess)  # 2 (bool)
NUM_IS_LEGENDARY = 2
NUM_IS_MYTHICAL = 2
NUM_IS_BABY = 2
NAME_MAX_LEN = 16
NAME_CHAR_PAD = 0
NAME_CHAR_VOCAB = 128

_DEFAULT_DATA_PATH = Path(__file__).parents[4] / "data" / "pokemon.json"


def get_name_chars(name: str) -> list[int]:
    stem = re.sub(r"(_shiny)?(_female)?(_frame2)?(_flip)?\.txt$", "", name)
    chars = [ord(c) for c in stem[:NAME_MAX_LEN]]
    return chars + [NAME_CHAR_PAD] * (NAME_MAX_LEN - len(chars))


def _pokemon_id(name: str) -> str:
    stem = name.replace(".txt", "")
    m = re.match(r"^(\d+)", stem)
    if not m:
        return ""
    return str(int(m.group(1)))


class MetadataAdapter:
    def __init__(
        self,
        data_path: Path | str = _DEFAULT_DATA_PATH,
    ):
        with open(data_path) as f:
            self._data: dict[str, dict] = {
                str(entry["id"]): entry for entry in json.load(f)
            }

    def get_pokemon_metadata(
        self,
        name: str,
        generation: str,
    ) -> dict:
        pid = _pokemon_id(name)
        entry = self._data.get(pid, {})

        type1_str = entry.get("type_1", "")
        type2_str = entry.get("type_2")
        evo_stage = entry.get("evolution_stage", 1)

        return {
            "type1_idx": TYPE_TO_IDX.get(type1_str, TYPE1_UNK),
            "type2_idx": TYPE_TO_IDX.get(type2_str, TYPE2_NONE)
            if type2_str
            else TYPE2_NONE,
            "is_shiny": int("_shiny" in name),
            "generation_idx": int(generation[3:]) - 1,
            "evolution_stage_idx": min(evo_stage - 1, NUM_EVO_STAGES - 1),
            "has_evolution_idx": int(entry.get("has_evolution", False)),
            "is_legendary_idx": int(entry.get("is_legendary", False)),
            "is_mythical_idx": int(entry.get("is_mythical", False)),
            "is_baby_idx": int(entry.get("is_baby", False)),
            "color_idx": COLOR_TO_IDX.get(entry.get("color", ""), COLOR_UNK),
            "shape_idx": SHAPE_TO_IDX.get(entry.get("shape", ""), SHAPE_UNK),
            "habitat_idx": HABITAT_TO_IDX.get(
                entry.get("habitat"),
                HABITAT_NONE,
            )
            if entry.get("habitat")
            else HABITAT_NONE,
            "name_chars": get_name_chars(name),
        }
