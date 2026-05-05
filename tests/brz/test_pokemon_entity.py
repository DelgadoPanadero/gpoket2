import numpy as np
import pytest
from src.domain.brz.pokemon import PokemonEntity


def make_entity(h=32, w=32, name="bulbasaur.png"):
    return PokemonEntity(
        name=name,
        generation="gen1",
        game_name="red-blue",
        image=np.zeros((h, w, 3), dtype=np.uint8),
    )


class TestDim:
    def test_returns_height_width_tuple(self):
        assert make_entity(32, 48).dim == (32, 48)

    def test_square_sprite(self):
        assert make_entity(64, 64).dim == (64, 64)

    def test_single_pixel(self):
        assert make_entity(1, 1).dim == (1, 1)


class TestSerialization:
    def test_image_serialized_as_nested_list(self):
        data = make_entity(2, 2).model_dump()
        assert isinstance(data["image"], list)
        assert isinstance(data["image"][0], list)
        assert isinstance(data["image"][0][0], list)

    def test_name_generation_game_name_in_dump(self):
        entity = make_entity(name="charmander.png")
        data = entity.model_dump()
        assert data["name"] == "charmander.png"
        assert data["generation"] == "gen1"
        assert data["game_name"] == "red-blue"
