import torch
from transformers import DataCollatorForLanguageModeling

CONDITIONING_FIELDS = [
    "type1_idx",
    "type2_idx",
    "is_shiny",
    "generation_idx",
    "evolution_stage_idx",
    "has_evolution_idx",
]

_DISCARD_FIELDS = [
    "name",
    "generation",
    "game_name",
    "chunk",
    "input_text",
    "original_text",
]


class ConditionedDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        conditioning = {
            field: torch.tensor(
                [f.pop(field) for f in features], dtype=torch.long
            )
            for field in CONDITIONING_FIELDS
        }
        name_chars = torch.tensor(
            [f.pop("name_chars") for f in features], dtype=torch.long
        )
        for f in features:
            for field in _DISCARD_FIELDS:
                f.pop(field, None)
        batch = super().__call__(features)
        batch["type1"] = conditioning["type1_idx"]
        batch["type2"] = conditioning["type2_idx"]
        batch["is_shiny"] = conditioning["is_shiny"]
        batch["generation"] = conditioning["generation_idx"]
        batch["evolution_stage"] = conditioning["evolution_stage_idx"]
        batch["has_evolution"] = conditioning["has_evolution_idx"]
        batch["name_chars"] = name_chars
        return batch
