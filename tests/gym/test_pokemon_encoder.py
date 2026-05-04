import numpy as np
import pytest
from src.application.slv.pokedex.encoder.pokemon_encoder import PokemonEncoder


@pytest.fixture
def encoder():
    return PokemonEncoder()


def pixel(b, g, r):
    return np.array([[[b, g, r]]], dtype=np.uint8)


class TestEncode:
    def test_blank_pixel_encodes_to_tilde(self, encoder):
        assert encoder._encode(pixel(255, 255, 255))[0][0] == "~"
        assert encoder._encode(pixel(9, 9, 9))[0][0] == "~"

    def test_color_pixel_encodes_to_correct_char(self, encoder):
        # BGR=[0, 0, 64]: r=1, g=0, b=0 → chr(1*16 + 59) = chr(75)
        assert encoder._encode(pixel(0, 0, 64))[0][0] == chr(75)

    def test_output_dimensions_match_image_shape(self, encoder):
        image = np.full((5, 7, 3), 128, dtype=np.uint8)
        result = encoder._encode(image)
        assert len(result) == 5
        assert all(len(row) == 7 for row in result)


class TestDecode:
    def test_tilde_decodes_to_white(self, encoder):
        image = encoder._decode([["~"]])
        assert image[0, 0].tolist() == [255, 255, 255]

    def test_roundtrip_preserves_pixel(self, encoder):
        original = pixel(64, 128, 192)
        char = encoder._encode(original)[0][0]
        decoded = encoder._decode([[char]])
        # quantized to nearest step of 64, centered at +32
        np.testing.assert_allclose(original[0, 0], decoded[0, 0], atol=32)


class TestTextConversion:
    def test_array_to_text_and_back_roundtrip(self, encoder):
        array = [["a", "b", "~"], ["c", "d", "e"]]
        assert encoder._text_to_array(encoder._array_to_text(array)) == array
