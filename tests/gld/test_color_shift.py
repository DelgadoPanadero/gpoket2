import pytest
from src.application.gld.prof_oak_pc.augmentation import ColorShift
from src.domain.slv.pokedex import PokedexEntity


def pixel_char(r, g, b):
    return chr(r * 16 + g * 4 + b + 59)


def make_data(char, rows=4, cols=4):
    return "\n".join([" ".join([char] * cols)] * rows)


def make_entity(data, name="poke.txt"):
    return PokedexEntity(
        name=name, generation="gen3", game_name="firered", data=data
    )


class TestColorShift:
    def test_returns_five_variants(self):
        entity = make_entity(make_data(pixel_char(1, 0, 0)))
        assert len(ColorShift().run(entity)) == 5

    def test_names_use_correct_suffixes(self):
        entity = make_entity(make_data(pixel_char(1, 0, 0)), name="poke.txt")
        results = ColorShift().run(entity)
        for result, suffix in zip(results, ColorShift.SUFFIXES):
            assert result.name == f"poke{suffix}.txt"

    def test_generation_and_game_name_preserved(self):
        entity = make_entity(make_data(pixel_char(1, 2, 3)))
        for result in ColorShift().run(entity):
            assert result.generation == entity.generation
            assert result.game_name == entity.game_name

    def test_blank_tiles_unchanged(self):
        data = make_data("~")
        entity = make_entity(data)
        for result in ColorShift().run(entity):
            assert result.data == data

    def test_rg_swap_exchanges_r_and_g_channels(self):
        # (r=2, g=1, b=0) → RG swap → (r=1, g=2, b=0)
        original = pixel_char(2, 1, 0)
        expected = pixel_char(1, 2, 0)
        entity = make_entity(make_data(original))
        rg_result = ColorShift().run(entity)[0]
        assert expected in rg_result.data
        assert original not in rg_result.data

    def test_rg_swap_is_its_own_inverse(self):
        data = make_data(pixel_char(3, 1, 2))
        entity = make_entity(data)
        cs = ColorShift()
        once = make_entity(cs.run(entity)[0].data)
        twice = cs.run(once)[0]
        assert twice.data == data

    def test_result_entities_are_valid_pokedex_entities(self):
        entity = make_entity(make_data(pixel_char(1, 2, 3)))
        for result in ColorShift().run(entity):
            assert isinstance(result, PokedexEntity)
            rows = result.data.split("\n")
            width = len(rows[0].split())
            assert all(len(r.split()) == width for r in rows)
