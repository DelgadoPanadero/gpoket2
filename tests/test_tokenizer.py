import pytest
pytest.importorskip("datasets")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")
from src.application.gld.prof_oak_pc.tokenizer import Pokenizer
from src.domain.slv.pokedex import PokedexEntity


# --- helpers ---

def blank_row():
    return " ".join(["~"] * 64)


def non_blank_row(char="a"):
    return " ".join([char] * 64)


def sprite(non_blank_rows=2, blank_rows=2):
    rows = [non_blank_row()] * non_blank_rows + [blank_row()] * blank_rows
    return "\n".join(rows)


def make_entity(name, data=None):
    return PokedexEntity(name=name, data=data or sprite())


# --- _clean_text ---

class TestCleanText:
    def test_blank_rows_are_removed(self):
        data = "\n".join([non_blank_row(), blank_row(), non_blank_row()])
        result = Pokenizer()._clean_text(data)
        tokens = result.split()
        # 2 non-blank rows × 64 tokens each = 128 tokens
        assert len(tokens) == 128

    def test_all_blank_sprite_returns_empty(self):
        data = "\n".join([blank_row()] * 4)
        result = Pokenizer()._clean_text(data)
        assert result.strip() == ""

    def test_all_non_blank_sprite_keeps_all_rows(self):
        n_rows = 3
        data = "\n".join([non_blank_row()] * n_rows)
        result = Pokenizer()._clean_text(data)
        tokens = result.split()
        assert len(tokens) == n_rows * 64

    def test_row_number_replaces_first_pixel(self):
        data = non_blank_row()
        result = Pokenizer()._clean_text(data)
        tokens = result.split()
        # First token should be row number "00"
        assert tokens[0] == "00"
        # Total still 64 tokens
        assert len(tokens) == 64

    def test_single_non_blank_row_has_63_pixels_plus_row_number(self):
        data = non_blank_row("b")
        result = Pokenizer()._clean_text(data)
        tokens = result.split()
        assert tokens[0] == "00"
        assert all(t == "b" for t in tokens[1:])
        assert len(tokens) == 64

    def test_mixed_row_is_not_blank(self):
        # A row with one non-~ char should not be filtered
        mixed = "a " + " ".join(["~"] * 63)
        data = mixed
        result = Pokenizer()._clean_text(data)
        assert result.strip() != ""


# --- tokenize: name_to_idx mapping ---

class TestNameToIdx:
    def _build_pokenizer_and_tokenize(self, entities):
        p = Pokenizer()
        p.train(entities)
        p.tokenize(entities)
        return p

    def test_num_pokemon_equals_unique_names(self):
        entities = [make_entity("a.txt"), make_entity("b.txt"), make_entity("c.txt")]
        p = self._build_pokenizer_and_tokenize(entities)
        assert p.num_pokemon == 3

    def test_duplicate_names_count_once(self):
        entities = [make_entity("a.txt"), make_entity("a.txt"), make_entity("b.txt")]
        p = self._build_pokenizer_and_tokenize(entities)
        assert p.num_pokemon == 2

    def test_same_name_same_index(self):
        entities = [make_entity("x.txt"), make_entity("y.txt")]
        p = self._build_pokenizer_and_tokenize(entities)
        assert p.name_to_idx["x.txt"] == p.name_to_idx["x.txt"]

    def test_different_names_different_indices(self):
        entities = [make_entity("a.txt"), make_entity("b.txt")]
        p = self._build_pokenizer_and_tokenize(entities)
        assert p.name_to_idx["a.txt"] != p.name_to_idx["b.txt"]

    def test_indices_are_zero_based_contiguous(self):
        entities = [make_entity(f"{c}.txt") for c in "abcd"]
        p = self._build_pokenizer_and_tokenize(entities)
        assert set(p.name_to_idx.values()) == {0, 1, 2, 3}

    def test_index_ordering_is_alphabetical(self):
        # name_to_idx is built from sorted(set(names))
        entities = [make_entity("z.txt"), make_entity("a.txt"), make_entity("m.txt")]
        p = self._build_pokenizer_and_tokenize(entities)
        assert p.name_to_idx["a.txt"] < p.name_to_idx["m.txt"] < p.name_to_idx["z.txt"]

    def test_dataset_contains_pokemon_idx_field(self):
        entities = [make_entity("a.txt"), make_entity("b.txt")]
        p = Pokenizer()
        p.train(entities)
        dataset = p.tokenize(entities)
        assert "pokemon_idx" in dataset["train"].column_names

    def test_pokemon_idx_values_in_dataset_are_valid(self):
        entities = [make_entity("a.txt"), make_entity("b.txt")]
        p = Pokenizer()
        p.train(entities)
        dataset = p.tokenize(entities)
        indices = dataset["train"]["pokemon_idx"]
        assert all(0 <= idx < p.num_pokemon for idx in indices)

    def test_single_pokemon_gets_index_zero(self):
        entities = [make_entity("solo.txt")]
        p = self._build_pokenizer_and_tokenize(entities)
        assert p.name_to_idx["solo.txt"] == 0
        assert p.num_pokemon == 1
