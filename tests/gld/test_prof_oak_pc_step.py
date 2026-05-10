import pytest

pytest.importorskip("datasets")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from unittest.mock import MagicMock
from src.application.gld.prof_oak_pc import ProfOakPcStep
from src.domain.slv.pokedex import PokedexEntity


def non_blank_row():
    return " ".join(["a"] * 64)


def sprite_64x64():
    return "\n".join([non_blank_row()] * 64)


def make_entity(name):
    return PokedexEntity(
        name=name,
        generation="gen3",
        game_name="firered",
        data=sprite_64x64(),
    )


def make_oversized_entity(name="large.txt"):
    return PokedexEntity(
        name=name,
        generation="gen3",
        game_name="firered",
        data="\n".join([" ".join(["a"] * 65)] * 65),
    )


def make_step(entities, context_length=1024):
    pokedex_repo = MagicMock()
    profoakpc_repo = MagicMock()
    profoakpc_repo.partition = "test"
    profoakpc_repo.save.return_value = "box-test"
    pokedex_repo.load_all.return_value = entities
    step = ProfOakPcStep(
        pokedex_repository=pokedex_repo,
        profoakpc_repository=profoakpc_repo,
        context_length=context_length,
    )
    return step, pokedex_repo, profoakpc_repo


class TestProfOakPcStep:
    def test_run_returns_saved_box_name(self):
        step, _, _ = make_step([make_entity("poke.txt")])
        assert step.run() == ["box-test"]

    def test_loads_from_pokedex_repository(self):
        step, pokedex_repo, _ = make_step([make_entity("poke.txt")])
        step.run()
        pokedex_repo.load_all.assert_called_once()

    def test_saves_box_to_profoakpc_repository(self):
        step, _, profoakpc_repo = make_step([make_entity("poke.txt")])
        step.run()
        profoakpc_repo.save.assert_called_once()

    def test_oversized_entities_filtered_out(self):
        step, _, profoakpc_repo = make_step([make_oversized_entity()])
        step.run()
        saved_box = profoakpc_repo.save.call_args[0][0]
        assert saved_box.dataset["train"].num_rows == 0

    def test_flip_augmentation_doubles_entities(self):
        entities = [make_entity(f"poke{i}.txt") for i in range(2)]
        step, _, profoakpc_repo = make_step(entities, context_length=1024)
        step.run()
        saved_box = profoakpc_repo.save.call_args[0][0]
        # 2 entities × 2 (flip) × 4 chunks (4096 tokens / 1024 ctx) = 16
        assert saved_box.dataset["train"].num_rows == 16

    def test_box_name_uses_repository_partition(self):
        step, _, profoakpc_repo = make_step([make_entity("poke.txt")])
        step.run()
        saved_box = profoakpc_repo.save.call_args[0][0]
        assert saved_box.name == "box-test"

    def test_dataset_has_required_columns(self):
        step, _, profoakpc_repo = make_step([make_entity("poke.txt")])
        step.run()
        saved_box = profoakpc_repo.save.call_args[0][0]
        cols = saved_box.dataset["train"].column_names
        for field in ("name", "input_ids", "labels", "pokemon_idx"):
            assert field in cols
