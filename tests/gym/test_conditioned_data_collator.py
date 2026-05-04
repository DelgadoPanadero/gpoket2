import pytest
from unittest.mock import MagicMock, patch

torch = pytest.importorskip("torch")
from src.application.gym.model.conditioned_data_collator import ConditionedDataCollator

PATCH = "src.application.gym.model.conditioned_data_collator.DataCollatorForLanguageModeling.__call__"


def make_collator():
    return ConditionedDataCollator(tokenizer=MagicMock(), mlm=False)


def parent_return(batch_size=1, seq_len=4):
    return {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


def make_features(pokemon_indices):
    return [
        {"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1], "pokemon_idx": idx}
        for idx in pokemon_indices
    ]


def test_pokemon_idx_extracted_correctly():
    with patch(PATCH, return_value=parent_return(3)):
        batch = make_collator()(make_features([0, 7, 3]))
    assert batch["pokemon_idx"].tolist() == [0, 7, 3]
    assert batch["pokemon_idx"].dtype == torch.long


def test_pokemon_idx_not_passed_to_parent_and_standard_fields_preserved():
    captured = []

    def fake_parent(self, features):
        captured.extend(features)
        return parent_return(len(features))

    with patch(PATCH, fake_parent):
        batch = make_collator()(make_features([1, 2]))

    assert all("pokemon_idx" not in f for f in captured)
    assert "input_ids" in batch and "labels" in batch and "attention_mask" in batch
