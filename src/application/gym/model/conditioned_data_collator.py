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
        pokemon_idx = torch.tensor(
            [f.pop("pokemon_idx") for f in features], dtype=torch.long
        )
        row_ids_vals = [f.pop("row_ids", None) for f in features]
        conditioning = {
            field: [f.pop(field, None) for f in features]
            for field in CONDITIONING_FIELDS
        }
        name_chars_vals = [f.pop("name_chars", None) for f in features]
        for f in features:
            for field in _DISCARD_FIELDS:
                f.pop(field, None)

        batch = super().__call__(features)
        batch["pokemon_idx"] = pokemon_idx

        if all(v is not None for v in row_ids_vals):
            batch["row_ids"] = torch.tensor(row_ids_vals, dtype=torch.long)

        if all(v is not None for v in conditioning["type1_idx"]):
            batch["type1"] = torch.tensor(conditioning["type1_idx"], dtype=torch.long)
            batch["type2"] = torch.tensor(conditioning["type2_idx"], dtype=torch.long)
            batch["is_shiny"] = torch.tensor(conditioning["is_shiny"], dtype=torch.long)
            batch["generation"] = torch.tensor(conditioning["generation_idx"], dtype=torch.long)
            batch["evolution_stage"] = torch.tensor(conditioning["evolution_stage_idx"], dtype=torch.long)
            batch["has_evolution"] = torch.tensor(conditioning["has_evolution_idx"], dtype=torch.long)

        if all(v is not None for v in name_chars_vals):
            batch["name_chars"] = torch.tensor(name_chars_vals, dtype=torch.long)

        return batch
