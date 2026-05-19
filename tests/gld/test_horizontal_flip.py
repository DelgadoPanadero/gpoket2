import pytest
from src.application.gld.prof_oak_pc.augmentation import HorizontalFlip
from src.domain.slv.pokedex import PokedexEntity


def make_entity(data, name="poke.txt"):
    return PokedexEntity(
        name=name, generation="gen3", game_name="firered", data=data
    )


def make_data(rows=4, cols=4):
    chars = list("abcd")
    return "\n".join(
        [" ".join([chars[i % len(chars)]] * cols) for i in range(rows)]
    )


class TestHorizontalFlip:
    def test_row_tokens_are_reversed(self):
        entity = make_entity("a b c d\ne f g h\ni j k l\nm n o p")
        result = HorizontalFlip().run(entity)
        assert result.data.split("\n")[0].split() == ["d", "c", "b", "a"]

    def test_all_rows_are_reversed(self):
        entity = make_entity("a b c\nd e f\ng h i")
        result = HorizontalFlip().run(entity)
        for original, flipped in zip(
            entity.data.split("\n"), result.data.split("\n")
        ):
            assert flipped.split() == original.split()[::-1]

    def test_name_gets_flip_suffix(self):
        entity = make_entity(make_data(), name="bulbasaur.txt")
        assert HorizontalFlip().run(entity).name == "bulbasaur_flip.txt"

    def test_generation_and_game_name_preserved(self):
        entity = make_entity(make_data())
        result = HorizontalFlip().run(entity)
        assert result.generation == entity.generation
        assert result.game_name == entity.game_name

    def test_double_flip_restores_original_data(self):
        entity = make_entity(make_data())
        flipper = HorizontalFlip()
        assert flipper.run(flipper.run(entity)).data == entity.data

    def test_horizontally_symmetric_sprite_unchanged(self):
        data = "\n".join(["a b b a"] * 4)
        entity = make_entity(data)
        assert HorizontalFlip().run(entity).data == data

    def test_single_token_row_unchanged(self):
        data = "\n".join(["x"] * 4)
        entity = make_entity(data)
        assert HorizontalFlip().run(entity).data == data
