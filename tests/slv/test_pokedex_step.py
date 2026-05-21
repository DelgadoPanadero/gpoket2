import numpy as np
import pytest
from unittest.mock import MagicMock
from src.application.slv.pokedex import PokedexStep
from src.domain.brz.pokemon import PokemonEntity


def make_pokemon(name="poke.png", generation="gen3", h=64, w=64):
    image = np.full((h, w, 3), 128, dtype=np.uint8)
    image[0, :] = image[-1, :] = image[:, 0] = image[:, -1] = 255
    return PokemonEntity(
        name=name, generation=generation, game_name="firered", image=image
    )


def make_step(pokemon_list=None):
    pokemon_repo = MagicMock()
    pokedex_repo = MagicMock()
    pokemon_repo.load_all.return_value = pokemon_list or []
    pokedex_repo.save_all.side_effect = lambda lst: [p.name for p in lst]
    return PokedexStep(
        pokemon_repository=pokemon_repo, pokedex_repository=pokedex_repo
    ), pokedex_repo


def test_invalid_generation_raises():
    step, _ = make_step()
    with pytest.raises(ValueError, match="Invalid generation"):
        step.run(generations=["gen99"])


def test_valid_pokemon_encoded_and_saved():
    pokemon = make_pokemon("bulbasaur.png", "gen3")
    step, pokedex_repo = make_step([pokemon])
    result = step.run(generations=["gen3"])
    pokedex_repo.save_all.assert_called_once()
    saved = pokedex_repo.save_all.call_args[0][0]
    assert len(saved) == 1
    assert saved[0].name == "bulbasaur.txt"
