import datasets

class SilverPokemon(datasets.GeneratorBasedBuilder):
    """Silver layer: Cleans and standardizes the Bronze Pokemon data."""

    def _info(self):
        # The schema is now cleaned and properly typed
        return datasets.DatasetInfo(
            description="Cleaned Pokemon data with correct types and handled missing values.",
            features=datasets.Features({
                "name": datasets.Value("string"),
                "types": datasets.Sequence(datasets.Value("string")),
                "hp": datasets.Value("int32"),
                "attack": datasets.Value("int32"),
                "defense": datasets.Value("int32"),
                "speed": datasets.Value("int32"),
                "is_legendary": datasets.Value("bool"),
            }),
        )

    def _split_generators(self, dl_manager):
        # No new data files are needed; we will load the Bronze dataset
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

    def _generate_examples(self, **kwargs):
        """Loads the Bronze dataset, cleans it, and yields Silver records."""
        # Load the previous layer. This is the key to chaining!
        bronze_ds = datasets.load_dataset(path="../bronze_pokemon", split="train")

        for key, example in enumerate(bronze_ds):
            # 1. Clean data and handle missing values
            type2 = example.get("Type 2")
            types = [example["Type 1"]]
            if type2:
                types.append(type2)

            # 2. Correct data types and structure
            yield key, {
                "name": example["Name"],
                "types": types,
                "hp": int(example["HP"]),
                "attack": int(example["Attack"]),
                "defense": int(example["Defense"]),
                "speed": int(example["Speed"]),
                "is_legendary": example["Legendary"].lower() == "true",
            }