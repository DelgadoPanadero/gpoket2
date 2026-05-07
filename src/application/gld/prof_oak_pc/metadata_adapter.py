import re
import json
from pathlib import Path

TYPES = [
    "Normal",
    "Fire",
    "Water",
    "Electric",
    "Grass",
    "Ice",
    "Fighting",
    "Poison",
    "Ground",
    "Flying",
    "Psychic",
    "Bug",
    "Rock",
    "Ghost",
    "Dragon",
    "Dark",
    "Steel",
    "Fairy",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}
TYPE1_UNK = len(TYPES)  # 18
TYPE2_NONE = len(TYPES)  # 18 — pokemon sin tipo secundario
TYPE2_UNK = len(TYPES) + 1  # 19
NUM_TYPES1 = len(TYPES) + 1  # 19 (0-17 tipos + UNK)
NUM_TYPES2 = len(TYPES) + 2  # 20 (0-17 tipos + NONE + UNK)
NUM_GENERATIONS = 10  # gen1-gen9 + margen
NUM_EVO_STAGES = 4  # stage 1/2/3 + UNK(3)
NUM_HAS_EVOLUTION = 2  # 0=no, 1=yes
NAME_MAX_LEN = 16  # caracteres máximos del nombre
NAME_CHAR_PAD = 0  # índice de padding
NAME_CHAR_VOCAB = 128  # ASCII 0-127

_DATA_PATH = (
    Path(__file__).parents[4] / "data" / "metadata" / "pokemon_types.json"
)

with open(_DATA_PATH) as _f:
    _TYPES_DATA: dict = json.load(_f)


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


def get_metadata(name: str, generation: str) -> dict:
    pid = _pokemon_id(name)
    entry = _TYPES_DATA.get(pid, {})

    type1_str = entry.get("type1", "")
    type2_str = entry.get("type2")
    evo_stage = entry.get("evolution_stage", 1)
    has_evo = entry.get("has_evolution", False)

    return {
        "type1_idx": TYPE_TO_IDX.get(type1_str, TYPE1_UNK),
        "type2_idx": TYPE_TO_IDX.get(type2_str, TYPE2_NONE)
        if type2_str
        else TYPE2_NONE,
        "is_shiny": int("_shiny" in name),
        "generation_idx": int(generation[3:]) - 1,
        "evolution_stage_idx": min(evo_stage - 1, NUM_EVO_STAGES - 1),
        "has_evolution_idx": int(has_evo),
        "name_chars": get_name_chars(name),
    }
