import datasets

class GoldPokemon(datasets.GeneratorBasedBuilder):
    """Gold layer: Creates model-ready features from the Silver data."""

    def _info(self):
        # The schema is now feature-engineered for a specific modeling task
        return datasets.DatasetInfo(
            description="Feature-engineered Pokemon data to predict 'is_legendary'.",
            features=datasets.Features({
                "features": datasets.Sequence(feature=datasets.Value("float32"), length=3),
                "label": datasets.ClassLabel(names=["Not Legendary", "Legendary"]),
            }),
        )

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

    def _generate_examples(self, **kwargs):
        """Loads the Silver dataset and generates features and labels."""
        silver_ds = datasets.load_dataset(path="../silver_pokemon", split="train")

        for key, example in enumerate(silver_ds):
            # Create a simple feature vector (in a real scenario, this would involve normalization, etc.)
            features = [
                float(example["hp"]),
                float(example["attack"]),
                float(example["defense"]),
            ]

            yield key, {
                "features": features,
                "label": example["is_legendary"],
            }