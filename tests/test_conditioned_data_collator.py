import pytest
from unittest.mock import MagicMock, patch

torch = pytest.importorskip("torch")
from src.application.train.conditioned_data_collator import ConditionedDataCollator

PATCH = "src.application.train.conditioned_data_collator.DataCollatorForLanguageModeling.__call__"


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


# --- pokemon_idx extraction ---

def test_pokemon_idx_present_in_output():
    with patch(PATCH, return_value=parent_return(1)):
        batch = make_collator()(make_features([5]))
    assert "pokemon_idx" in batch


def test_pokemon_idx_correct_values():
    with patch(PATCH, return_value=parent_return(3)):
        batch = make_collator()(make_features([0, 7, 3]))
    assert batch["pokemon_idx"].tolist() == [0, 7, 3]


def test_pokemon_idx_dtype_is_long():
    with patch(PATCH, return_value=parent_return(2)):
        batch = make_collator()(make_features([1, 2]))
    assert batch["pokemon_idx"].dtype == torch.long


def test_pokemon_idx_shape_single_sample():
    with patch(PATCH, return_value=parent_return(1)):
        batch = make_collator()(make_features([42]))
    assert batch["pokemon_idx"].shape == torch.Size([1])


def test_pokemon_idx_shape_multiple_samples():
    n = 5
    with patch(PATCH, return_value=parent_return(n)):
        batch = make_collator()(make_features(list(range(n))))
    assert batch["pokemon_idx"].shape == torch.Size([n])


def test_pokemon_idx_zero_is_valid():
    with patch(PATCH, return_value=parent_return(1)):
        batch = make_collator()(make_features([0]))
    assert batch["pokemon_idx"].item() == 0


# --- pokemon_idx removed before parent call ---

def test_pokemon_idx_not_passed_to_parent():
    captured = []

    def fake_parent(features):
        captured.extend(features)
        return parent_return(len(features))

    with patch(PATCH, fake_parent):
        make_collator()(make_features([1, 2]))

    assert all("pokemon_idx" not in f for f in captured)


# --- standard fields unaffected ---

def test_standard_fields_still_present():
    with patch(PATCH, return_value=parent_return(2)):
        batch = make_collator()(make_features([0, 1]))
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
