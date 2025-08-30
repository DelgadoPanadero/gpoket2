import csv
import datasets

# Path to the raw data file relative to this script
_DATA_PATH = "data/pokemon.csv"

class BronzePokemon(datasets.GeneratorBasedBuilder):
    """Bronze layer: Ingests raw Pokemon data with minimal changes."""

    def _info(self):
        return datasets.DatasetInfo(
            description="Raw data of Pokemon stats and types.",
            features=datasets.Features({
                '#': datasets.Value("string"),
                'Name': datasets.Value("string"),
                'Type 1': datasets.Value("string"),
                'Type 2': datasets.Value("string"),
                'Total': datasets.Value("string"),
                'HP': datasets.Value("string"),
                'Attack': datasets.Value("string"),
                'Defense': datasets.Value("string"),
                'Sp. Atk': datasets.Value("string"),
                'Sp. Def': datasets.Value("string"),
                'Speed': datasets.Value("string"),
                'Generation': datasets.Value("string"),
                'Legendary': datasets.Value("string"),
            }),
        )

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": _DATA_PATH})]

    def _generate_examples(self, filepath):
        """Yields examples as-is from the CSV."""
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for key, row in enumerate(reader):
                yield key, row