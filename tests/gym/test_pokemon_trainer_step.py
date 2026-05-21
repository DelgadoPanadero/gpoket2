import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("transformers")

from datasets import Dataset, DatasetDict
from src.application.gym.train.pokemon_trainer_step import PokemonTrainerStep

PATCH_TRAINER = "src.application.gym.train.pokemon_trainer_step.Trainer"


def make_tokenizer():
    tok = MagicMock()
    tok.get_vocab.return_value = {f"t{i}": i for i in range(50)}
    tok.convert_tokens_to_ids.return_value = 0
    tok.bos_token_id = 1
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    return tok


def make_box():
    box = MagicMock()
    box.tokenizer = make_tokenizer()
    box.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "input_ids": [[1, 2]] * 4,
                    "labels": [[1, 2]] * 4,
                    "pokemon_idx": [0, 1, 0, 1],
                }
            ),
        }
    )
    return box


def make_step(**kwargs):
    profoakpc_repo = MagicMock()
    checkpoint_adapter = MagicMock()
    checkpoint_adapter.get_latest_checkpoint.return_value = None
    step = PokemonTrainerStep(
        profoakpc_repository=profoakpc_repo,
        checkpoint_storage_adapter=checkpoint_adapter,
        context_length=32,
        **kwargs,
    )
    return step, profoakpc_repo, checkpoint_adapter


@patch(PATCH_TRAINER)
def test_run_loads_box_and_calls_trainer(MockTrainer):
    step, profoakpc_repo, _ = make_step()
    profoakpc_repo.load.return_value = make_box()
    step.run()
    profoakpc_repo.load.assert_called_once()
    MockTrainer.return_value.train.assert_called_once()


@patch(PATCH_TRAINER)
def test_no_upload_when_model_repo_is_none(MockTrainer):
    step, profoakpc_repo, checkpoint_adapter = make_step(model_repository=None)
    profoakpc_repo.load.return_value = make_box()
    step.run()
    # get_latest_checkpoint is called once by _num_pokemon_from_checkpoint,
    # but never a second time for the upload path (model_repository is None)
    checkpoint_adapter.get_latest_checkpoint.assert_called_once()


@patch(PATCH_TRAINER)
def test_upload_called_when_model_repo_and_checkpoint(MockTrainer):
    model_repo = MagicMock()
    step, profoakpc_repo, checkpoint_adapter = make_step(
        model_repository=model_repo,
        hf_repo_id="org/repo",
        hf_version="v1",
    )
    profoakpc_repo.load.return_value = make_box()
    checkpoint_adapter.get_latest_checkpoint.return_value = "/ckpt/step-100"
    step.run()
    model_repo.upload.assert_called_once()
