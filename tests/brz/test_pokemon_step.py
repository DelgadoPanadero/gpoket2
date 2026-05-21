import numpy as np
from unittest.mock import MagicMock
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
    return PokemonStep(pokemon_repository=repo, pokemon_extraction_adapter=adapter), repo


def test_saves_one_per_sprite():
    entities = [make_entity("a.png"), make_entity("b.png"), make_entity("c.png")]
    step, repo = make_step(entities)
    step.run()
    assert repo.save_one.call_count == 3


def test_returns_paths_from_repository():
    entities = [make_entity("a.png"), make_entity("b.png")]
    step, _ = make_step(entities)
    assert step.run() == ["path/a.png", "path/b.png"]
