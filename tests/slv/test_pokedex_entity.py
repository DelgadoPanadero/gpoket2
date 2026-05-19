import pytest
from pydantic import ValidationError
from src.domain.slv.pokedex import PokedexEntity


def make_data(rows=4, cols=4, char="a"):
    return "\n".join([" ".join([char] * cols)] * rows)


def make_entity(**kwargs):
    defaults = dict(
        name="x.txt", generation="gen3", game_name="firered", data=make_data()
    )
    return PokedexEntity(**(defaults | kwargs))


class TestValidator:
    def test_uniform_grid_is_accepted(self):
        entity = make_entity(data=make_data(8, 8))
        assert entity.data == make_data(8, 8)

    def test_nonuniform_row_width_raises(self):
        with pytest.raises(ValidationError):
            make_entity(data="a b c\na b")

    def test_single_row_is_valid(self):
        make_entity(data="a b c d")

    def test_single_token_is_valid(self):
        make_entity(data="a")

    def test_blank_only_grid_is_valid(self):
        make_entity(data=make_data(4, 4, "~"))


class TestSize:
    def test_size_equals_number_of_rows(self):
        assert make_entity(data=make_data(rows=8, cols=4)).size == 8

    def test_size_one_for_single_row(self):
        assert make_entity(data=make_data(rows=1, cols=4)).size == 1

    def test_size_64_for_full_sprite(self):
        assert make_entity(data=make_data(rows=64, cols=64)).size == 64
