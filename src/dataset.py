import os

from pathlib import Path
from datasets import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


class PokeCenter:

    def __init__(
        self,
        model_dir,
        row_length=32,
        context_length=1024,
    ):

        self._row_length = row_length
        self._context_length = context_length
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

    def _tokenize(
        self,
        sample,
    ):
        """ """
        outputs = self._tokenizer(
            sample["text"],
            return_overflowing_tokens=True,
            return_length=True,
        )

        #input_batch = []
        #for size, ids in zip(outputs["length"], outputs["input_ids"]):
        #    if size != self._row_length+1:
        #        print(f"Missing Pokemon!!! Bad encoding size: {size}")
        #        return {"input_ids": []}
        #    input_batch += ids

        return outputs #{"input_ids": [input_batch]}


    def create_dataset(
        self,
        source_dir,
    ):
        """ """
        pokedex = load_dataset("./pokemons_txt")

        #paths = [str(x) for x in Path(source_dir).glob("**/*.txt")]

        #data = []
        #for file_path in paths:
        #    with open(file_path, "r", encoding="utf-8") as f:
        #        texto = f.read()
        #        data.append(texto)

        #pokedex = Dataset.from_dict({"text": data})
        pokedex = pokedex.map(
            self._tokenize,
            batched=True,
            batch_size=1,
            remove_columns=pokedex["train"].column_names,
        )

        pokedex.save_to_disk(source_dir)