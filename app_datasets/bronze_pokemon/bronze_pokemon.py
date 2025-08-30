import cv2
import datasets
import numpy as np
from pathlib import Path

# Path to the raw data directory relative to this script
_DATA_PATH = "data/brz/pokemon"

class BronzePokemon(datasets.GeneratorBasedBuilder):
    """Bronze layer: Ingests raw Pokemon image data with minimal changes."""

    def _info(self):
        return datasets.DatasetInfo(
            description="Raw Pokemon image data from the bronze layer.",
            features=datasets.Features({
                'name': datasets.Value("string"),
                'image': datasets.Array3D(shape=(None, None, 3), dtype="uint8"),
                'file_path': datasets.Value("string"),
            }),
        )

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"data_dir": _DATA_PATH})]

    def _generate_examples(self, data_dir):
        """Yields Pokemon image examples from the bronze data directory."""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory {data_path} does not exist")
        
        # Get all PNG files in the directory
        png_files = list(data_path.glob("*.png"))
        
        for key, img_path in enumerate(png_files):
            try:
                # Load image using OpenCV (same as in the original repository)
                image = cv2.imread(str(img_path))
                
                if image is not None:
                    # Convert BGR to RGB (OpenCV loads as BGR by default)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    yield key, {
                        'name': img_path.name,
                        'image': image_rgb,
                        'file_path': str(img_path),
                    }
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
