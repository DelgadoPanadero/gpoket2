import numpy as np
import pytest
from unittest.mock import MagicMock
from src.application.slv.pokedex import PokedexStep
from src.domain.brz.pokemon import PokemonEntity
from src.domain.slv.pokedex import PokedexEntity


def make_pokemon(name="poke.png", generation="gen3", h=64, w=64):
    return PokemonEntity(
        name=name,
        generation=generation,
        game_name="firered",
        image=np.zeros((h, w, 3), dtype=np.uint8),
    )


def make_step(pokemon_list=None):
    mock_pokemon_repo = MagicMock()
    mock_pokedex_repo = MagicMock()
    mock_pokemon_repo.load_all.return_value = pokemon_list or []
    mock_pokedex_repo.save_all.side_effect = lambda lst: [p.name for p in lst]
    step = PokedexStep(
        pokemon_repository=mock_pokemon_repo,
        pokedex_repository=mock_pokedex_repo,
    )
    return step, mock_pokemon_repo, mock_pokedex_repo


class TestGenerationValidation:
    def test_invalid_generation_raises_value_error(self):
        step, _, _ = make_step()
        with pytest.raises(ValueError, match="Invalid generation"):
            step.run(generations=["gen99"])

    def test_valid_and_invalid_mix_raises(self):
        step, _, _ = make_step()
        with pytest.raises(ValueError):
            step.run(generations=["gen3", "gen99"])

    def test_all_valid_generations_accepted(self):
        step, _, _ = make_step()
        for gen in ["gen1", "gen2", "gen3", "gen4"]:
            step.run(generations=[gen])


class TestRunBehavior:
    def test_loads_each_requested_generation(self):
        step, pokemon_repo, _ = make_step()
        step.run(generations=["gen3", "gen4"])
        assert pokemon_repo.load_all.call_count == 2

    def test_encodes_pokemon_and_saves_pokedex(self):
        pokemon = make_pokemon("bulbasaur.png", "gen3")
        step, _, pokedex_repo = make_step([pokemon])
        result = step.run(generations=["gen3"])
        pokedex_repo.save_all.assert_called_once()
        saved = pokedex_repo.save_all.call_args[0][0]
        assert len(saved) == 1
        assert saved[0].name == "bulbasaur.txt"

    def test_returns_paths_from_repository(self):
        pokemon = make_pokemon("squirtle.png", "gen3")
        step, _, pokedex_repo = make_step([pokemon])
        pokedex_repo.save_all.return_value = ["squirtle.txt"]
        result = step.run(generations=["gen3"])
        assert result == ["squirtle.txt"]

    def test_empty_pokemon_list_saves_empty(self):
        step, _, pokedex_repo = make_step([])
        step.run(generations=["gen3"])
        saved = pokedex_repo.save_all.call_args[0][0]
        assert saved == []
