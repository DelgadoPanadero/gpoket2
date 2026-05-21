import pytest

pytest.importorskip("datasets")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from unittest.mock import MagicMock
from src.application.gld.prof_oak_pc import ProfOakPcStep
from src.domain.slv.pokedex import PokedexEntity


def sprite_64x64():
    row = " ".join(["a"] * 64)
    return "\n".join([row] * 64)


def make_entity(name="poke.txt"):
    return PokedexEntity(name=name, generation="gen3", game_name="firered", data=sprite_64x64())


def make_oversized_entity(name="large.txt"):
    row = " ".join(["a"] * 65)
    return PokedexEntity(name=name, generation="gen3", game_name="firered", data="\n".join([row] * 65))


def make_step(entities, context_length=5000):
    pokedex_repo = MagicMock()
    profoakpc_repo = MagicMock()
    profoakpc_repo.partition = "test"
    profoakpc_repo.save.return_value = "box-test"
    pokedex_repo.load_all.return_value = entities
    return ProfOakPcStep(
        pokedex_repository=pokedex_repo,
        profoakpc_repository=profoakpc_repo,
        context_length=context_length,
    ), profoakpc_repo


def test_run_returns_box_name():
    step, _ = make_step([make_entity()])
    assert step.run() == ["box-test"]


def test_oversized_entities_filtered_out():
    step, profoakpc_repo = make_step([make_oversized_entity()])
    step.run()
    saved_box = profoakpc_repo.save.call_args[0][0]
    assert saved_box.dataset["train"].num_rows == 0


def test_dataset_has_required_columns():
    step, profoakpc_repo = make_step([make_entity()])
    step.run()
    saved_box = profoakpc_repo.save.call_args[0][0]
    cols = saved_box.dataset["train"].column_names
    for field in ("name", "input_ids", "labels", "pokemon_idx"):
        assert field in cols
