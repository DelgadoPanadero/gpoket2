import numpy as np
import pytest
from unittest.mock import MagicMock, call
from src.application.brz.pokemon import PokemonStep
from src.domain.brz.pokemon import PokemonEntity


def make_entity(name="poke.png"):
    return PokemonEntity(
        name=name,
        generation="gen1",
        game_name="red-blue",
        image=np.zeros((32, 32, 3), dtype=np.uint8),
    )


def make_step(entities):
    adapter = MagicMock()
    repo = MagicMock()
    adapter.extract_sprites.return_value = entities
    repo.save_one.side_effect = [f"path/{e.name}" for e in entities]
    return PokemonStep(
        pokemon_repository=repo, pokemon_extraction_adapter=adapter
    ), repo


class TestPokemonStep:
    def test_saves_each_sprite_once(self):
        entities = [
            make_entity("a.png"),
            make_entity("b.png"),
            make_entity("c.png"),
        ]
        step, repo = make_step(entities)
        step.run()
        assert repo.save_one.call_count == 3

    def test_returns_paths_from_repository(self):
        entities = [make_entity("a.png"), make_entity("b.png")]
        step, _ = make_step(entities)
        result = step.run()
        assert result == ["path/a.png", "path/b.png"]

    def test_empty_adapter_returns_empty_list(self):
        step, repo = make_step([])
        assert step.run() == []
        repo.save_one.assert_not_called()

    def test_each_entity_passed_to_save(self):
        entities = [make_entity("a.png"), make_entity("b.png")]
        step, repo = make_step(entities)
        step.run()
        saved = [c.args[0] for c in repo.save_one.call_args_list]
        assert saved == entities
