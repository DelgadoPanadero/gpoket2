import numpy as np
import pytest
from src.application.slv.pokedex.pokemon_encoder import PokemonEncoder


@pytest.fixture
def encoder():
    return PokemonEncoder()


def pixel(b, g, r):
    return np.array([[[b, g, r]]], dtype=np.uint8)


# --- blank detection ---


class TestBlankDetection:
    def test_white_pixel_is_blank(self, encoder):
        assert encoder._encode(pixel(255, 255, 255))[0][0] == "~"

    def test_threshold_246_is_blank(self, encoder):
        # min > 245 → blank
        assert encoder._encode(pixel(246, 246, 246))[0][0] == "~"

    def test_threshold_245_is_not_blank(self, encoder):
        # min = 245, not > 245 → not blank by near-white rule
        # max = 245, not < 10 → not blank by near-black rule
        assert encoder._encode(pixel(245, 245, 245))[0][0] != "~"

    def test_near_black_9_is_blank(self, encoder):
        # max < 10 → blank
        assert encoder._encode(pixel(9, 9, 9))[0][0] == "~"

    def test_threshold_10_is_not_blank(self, encoder):
        # max = 10, not < 10 → not blank by near-black rule
        # min = 10, not > 245 → not blank by near-white rule
        assert encoder._encode(pixel(10, 10, 10))[0][0] != "~"

    def test_mixed_channel_near_white_is_blank(self, encoder):
        # min > 245 even if channels differ
        assert encoder._encode(pixel(250, 252, 248))[0][0] == "~"

    def test_mixed_channel_one_below_threshold_not_blank(self, encoder):
        # min = 244, not > 245 → near-white rule fails
        # max = 252, not < 10 → near-black rule fails → not blank
        assert encoder._encode(pixel(244, 252, 248))[0][0] != "~"

    def test_all_white_image_all_blank(self, encoder):
        image = np.full((4, 4, 3), 255, dtype=np.uint8)
        result = encoder._encode(image)
        assert all(result[r][c] == "~" for r in range(4) for c in range(4))


# --- BGR channel order ---


class TestBGROrder:
    def test_blue_only_channel(self, encoder):
        # BGR = [64, 0, 0]: b=64//64=1, g=0, r=0
        # char = chr(r*16 + g*4 + b + 59) = chr(0 + 0 + 1 + 59) = chr(60)
        result = encoder._encode(pixel(64, 0, 0))[0][0]
        assert result == chr(60)

    def test_red_only_channel(self, encoder):
        # BGR = [0, 0, 64]: b=0, g=0, r=64//64=1
        # char = chr(1*16 + 0 + 0 + 59) = chr(75)
        result = encoder._encode(pixel(0, 0, 64))[0][0]
        assert result == chr(75)

    def test_green_only_channel(self, encoder):
        # BGR = [0, 64, 0]: b=0, g=64//64=1, r=0
        # char = chr(0 + 1*4 + 0 + 59) = chr(63)
        result = encoder._encode(pixel(0, 64, 0))[0][0]
        assert result == chr(63)

    def test_channel_quantization(self, encoder):
        # Values 0-63 → 0, 64-127 → 1, 128-191 → 2, 192-255 → 3
        r1 = encoder._encode(pixel(63, 0, 0))[0][0]  # b=0
        r2 = encoder._encode(pixel(64, 0, 0))[0][0]  # b=1
        assert r1 != r2


# --- output structure ---


class TestOutputStructure:
    def test_output_dimensions_match_height_width(self, encoder):
        image = np.full((5, 7, 3), 128, dtype=np.uint8)
        result = encoder._encode(image)
        assert len(result) == 5
        assert all(len(row) == 7 for row in result)

    def test_all_characters_are_single_char(self, encoder):
        image = np.full((4, 4, 3), 128, dtype=np.uint8)
        result = encoder._encode(image)
        assert all(len(result[r][c]) == 1 for r in range(4) for c in range(4))

    def test_64x64_image_produces_64x64_array(self, encoder):
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = encoder._encode(image)
        assert len(result) == 64
        assert all(len(row) == 64 for row in result)


# --- character mapping ---


class TestCharacterMapping:
    def test_64_unique_non_blank_characters(self, encoder):
        chars = set()
        for r in range(4):
            for g in range(4):
                for b in range(4):
                    char = chr(r * 16 + g * 4 + b + 59)
                    chars.add(char)
        assert len(chars) == 64

    def test_tilde_not_in_character_range(self):
        # '~' is chr(126); verify it's not produced by the formula
        for r in range(4):
            for g in range(4):
                for b in range(4):
                    assert chr(r * 16 + g * 4 + b + 59) != "~"

    def test_array_to_text_format(self, encoder):
        array = [["a", "b"], ["c", "d"]]
        text = encoder._array_to_text(array)
        assert text == "a b\nc d"
