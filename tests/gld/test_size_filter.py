import pytest
from src.application.gld.prof_oak_pc.filter import SizeFilter
from src.domain.slv.pokedex import PokedexEntity


def make_data(rows, cols, char="a"):
    return "\n".join([" ".join([char] * cols)] * rows)


def make_entity(rows, cols, char="a", name="poke.txt"):
    return PokedexEntity(
        name=name,
        generation="gen3",
        game_name="firered",
        data=make_data(rows, cols, char),
    )


class TestRemoveBlankLines:
    def test_removes_all_blank_rows(self):
        data = "a b c\n~ ~ ~\na b c"
        result = SizeFilter().remove_blank_lines(data)
        rows = result.split("\n")
        assert len(rows) == 2
        assert all("~" not in row.split() for row in rows)

    def test_removes_all_blank_columns(self):
        data = "~ a ~\n~ b ~\n~ c ~"
        result = SizeFilter().remove_blank_lines(data)
        for row in result.split("\n"):
            assert "~" not in row.split()

    def test_all_blank_grid_returns_empty_or_single(self):
        data = "~ ~ ~\n~ ~ ~"
        result = SizeFilter().remove_blank_lines(data)
        assert result.strip() == ""

    def test_no_blank_rows_unchanged_dimensions(self):
        data = make_data(4, 4, "a")
        result = SizeFilter().remove_blank_lines(data)
        rows = result.split("\n")
        assert len(rows) == 4
        assert all(len(r.split()) == 4 for r in rows)


class TestPadWithBlanks:
    def test_output_has_correct_dimensions(self):
        f = SizeFilter(size=8)
        result = f.pad_with_blanks(make_data(4, 4))
        rows = result.split("\n")
        assert len(rows) == 8
        assert all(len(r.split()) == 8 for r in rows)

    def test_already_correct_size_unchanged(self):
        f = SizeFilter(size=4)
        data = make_data(4, 4, "a")
        result = f.pad_with_blanks(data)
        rows = result.split("\n")
        assert len(rows) == 4
        assert all(len(r.split()) == 4 for r in rows)

    def test_padding_tokens_are_blank(self):
        f = SizeFilter(size=6)
        result = f.pad_with_blanks(make_data(2, 2, "a"))
        rows = result.split("\n")
        assert rows[0] == " ".join(["~"] * 6)


class TestRun:
    def test_returns_none_if_content_too_tall(self):
        f = SizeFilter(size=8)
        entity = make_entity(10, 8)
        assert f.run(entity) is None

    def test_returns_none_if_content_too_wide(self):
        f = SizeFilter(size=8)
        entity = make_entity(8, 10)
        assert f.run(entity) is None

    def test_valid_sprite_returns_padded_entity(self):
        f = SizeFilter(size=8)
        result = f.run(make_entity(4, 4))
        assert result is not None
        rows = result.data.split("\n")
        assert len(rows) == 8
        assert all(len(r.split()) == 8 for r in rows)

    def test_preserves_metadata(self):
        f = SizeFilter(size=8)
        entity = make_entity(4, 4, name="charmander.txt")
        result = f.run(entity)
        assert result.name == "charmander.txt"
        assert result.generation == "gen3"
        assert result.game_name == "firered"

    def test_exact_size_passes_through(self):
        f = SizeFilter(size=4)
        entity = make_entity(4, 4)
        result = f.run(entity)
        assert result is not None

    def test_blank_border_removed_before_size_check(self):
        f = SizeFilter(size=4)
        inner = make_data(3, 3, "a")
        rows = inner.split("\n")
        padded = "\n".join(
            ["~ ~ ~ ~ ~"] + [f"~ {r} ~" for r in rows] + ["~ ~ ~ ~ ~"]
        )
        entity = PokedexEntity(
            name="x.txt", generation="gen3", game_name="firered", data=padded
        )
        result = f.run(entity)
        assert result is not None
