import datasets
import numpy as np

class SilverPokemon(datasets.GeneratorBasedBuilder):
    """Silver layer: Encodes Pokemon images to text using the PokemonEncoder logic."""
    
    # Constants matching the original Pokenizer
    BOL_TOKEN = "00"

    def _info(self):
        return datasets.DatasetInfo(
            description="Pokemon images encoded to text format using character encoding.",
            features=datasets.Features({
                'name': datasets.Value("string"),
                'encoded_data': datasets.Value("string"),
                'original_file_path': datasets.Value("string"),
            }),
        )

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

    def _generate_examples(self, **kwargs):
        """Loads the Bronze dataset, encodes images to text, and yields Silver records."""
        # Load the bronze layer dataset
        bronze_ds = datasets.load_dataset(path="../bronze_pokemon", split="train", streaming=True)

        for key, example in enumerate(bronze_ds):
            try:
                # Apply the encoding logic from PokemonEncoder
                encoded_text = self._encode_image_to_text(example['image'])
                
                # Add BOL tokens to match original Pokenizer logic
                encoded_text_with_bol = self._add_bol_tokens(encoded_text)
                
                # Convert .png name to .txt name (following the original logic)
                encoded_name = example['name'].replace('.png', '.txt')
                
                yield key, {
                    'name': encoded_name,
                    'encoded_data': encoded_text_with_bol,
                    'original_file_path': example['file_path'],
                }
            except Exception as e:
                print(f"Error encoding image {example.get('name', 'unknown')}: {e}")
                continue

    @staticmethod
    def _encode_image_to_text(image):
        """
        Encodes an image to text using the same logic as PokemonEncoder.
        This replicates the _encode and _array_to_text methods from the src folder.
        """
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        width, height, _ = image.shape
        
        array = []
        for y in range(height):
            row = []
            for x in range(width):
                r, g, b = image[y, x] // 64
                is_blank = min(image[y, x]) > 245 or max(image[y, x]) < 10
                char = (
                    "~"
                    if is_blank
                    else chr(r * 4**2 + g * 4**1 + b * 4**0 + 59)
                )
                row.append(char)
            array.append(row)
        
        # Convert array to text (same as _array_to_text method)
        return "\n".join([" ".join(r) for r in array])

    def _add_bol_tokens(self, encoded_text: str) -> str:
        """
        Add BOL tokens to match original Pokenizer logic.
        This replicates the _clean_text method from the src folder.
        """
        text_split = encoded_text.split("\n")
        text_split = [[self.BOL_TOKEN] + row.split() for row in text_split]
        return "\n".join([" ".join(row) for row in text_split])
