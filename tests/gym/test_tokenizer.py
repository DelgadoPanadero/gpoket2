import pytest

pytest.importorskip("datasets")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")
from src.application.gld.prof_oak_pc.tokenizer import Pokenizer
from src.domain.slv.pokedex import PokedexEntity


def blank_row():
    return " ".join(["~"] * 64)


def non_blank_row(char="a"):
    return " ".join([char] * 64)


def sprite(non_blank_rows=2, blank_rows=2):
    rows = [non_blank_row()] * non_blank_rows + [blank_row()] * blank_rows
    return "\n".join(rows)


def make_entity(name, data=None):
    return PokedexEntity(
        name=name,
        generation="gen1",
        game_name="red-blue",
        data=data or sprite(),
    )


class TestCleanText:
    def test_row_number_replaces_first_pixel(self):
        data = non_blank_row()
        tokens = Pokenizer()._clean_text(data).split()
        assert tokens[0] == "00"
        assert len(tokens) == 64  # 1 row number + 63 pixels

    def test_all_rows_concatenated_with_sequential_numbers(self):
        data = "\n".join([non_blank_row("a"), non_blank_row("b")])
        tokens = Pokenizer()._clean_text(data).split()
        assert len(tokens) == 128
        assert tokens[0] == "00"
        assert tokens[64] == "01"


class TestTokenize:
    def _pokenizer(self, entities):
        p = Pokenizer()
        p.train(entities)
        return p

    def test_output_has_required_fields(self):
        entities = [make_entity("a.txt"), make_entity("b.txt")]
        ds = self._pokenizer(entities).tokenize(entities)
        for field in ("name", "input_ids", "labels", "attention_mask", "pokemon_idx"):
            assert field in ds["train"].column_names

    def test_pokemon_idx_values_are_valid_and_unique_per_name(self):
        entities = [make_entity("a.txt"), make_entity("b.txt")]
        p = self._pokenizer(entities)
        ds = p.tokenize(entities)
        indices = ds["train"]["pokemon_idx"]
        assert all(0 <= idx < p.num_pokemon for idx in indices)
        assert p.name_to_idx["a.txt"] != p.name_to_idx["b.txt"]
