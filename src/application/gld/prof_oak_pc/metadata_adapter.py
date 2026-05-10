import re
import json
from pathlib import Path

TYPES = [
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}
TYPE1_UNK = len(TYPES)  # 18
TYPE2_NONE = len(TYPES)  # 18 — pokemon sin tipo secundario
TYPE2_UNK = len(TYPES) + 1  # 19
NUM_TYPES1 = len(TYPES) + 1  # 19 (0-17 tipos + UNK)
NUM_TYPES2 = len(TYPES) + 2  # 20 (0-17 tipos + NONE + UNK)

COLORS = [
    "red",
    "blue",
    "yellow",
    "green",
    "black",
    "brown",
    "purple",
    "gray",
    "white",
    "pink",
]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}
COLOR_UNK = len(COLORS)
NUM_COLORS = len(COLORS) + 1

SHAPES = [
    "ball",
    "squiggle",
    "fish",
    "arms",
    "blob",
    "upright",
    "legs",
    "quadruped",
    "wings",
    "tentacles",
    "heads",
    "humanoid",
    "bug-wings",
    "armor",
]
SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}
SHAPE_UNK = len(SHAPES)
NUM_SHAPES = len(SHAPES) + 1

HABITATS = [
    "cave",
    "forest",
    "grassland",
    "mountain",
    "rare",
    "rough-terrain",
    "sea",
    "urban",
    "waters-edge",
]
HABITAT_TO_IDX = {h: i for i, h in enumerate(HABITATS)}
HABITAT_NONE = len(HABITATS)
HABITAT_UNK = len(HABITATS) + 1
NUM_HABITATS = len(HABITATS) + 2

NUM_GENERATIONS = 10  # gen1-gen9 + margen
NUM_EVO_STAGES = 4  # stage 1/2/3 + UNK(3)
NUM_HAS_EVOLUTION = 2
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
                entry.get("habitat"), HABITAT_NONE
            )
            if entry.get("habitat")
            else HABITAT_NONE,
            "name_chars": get_name_chars(name),
        }
