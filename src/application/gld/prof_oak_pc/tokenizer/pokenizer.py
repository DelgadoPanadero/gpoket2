from typing import List

from datasets import Value
from datasets import Dataset
from datasets import DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast  # type: ignore
from tokenizers.pre_tokenizers import WhitespaceSplit

from src.domain.slv.pokedex import PokedexEntity


class Pokenizer:
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    PAD_TOKEN = "[PAD]"

    def __init__(
        self,
        row_length: int = 64,
        col_length: int = 64,
        context_length: int = 4096,
    ):
        self.row_length = row_length
        self.col_length = col_length
        self.context_length = context_length

        _tokenizer = Tokenizer(WordLevel())  # type: ignore
        _tokenizer.pre_tokenizer = WhitespaceSplit()  # type: ignore

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=_tokenizer,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            pad_token=self.PAD_TOKEN,  # GPT-2 no necesita padding; para el collator lo igualamos a eos
        )

    def _clean_text(
        self,
        text: str,
    ) -> str:
        text_split = text.split("\n")

        text_split = [
            ["%02d" % pos] + row.split()[1:]
            for pos, row in enumerate(text_split)
        ]

        return " ".join([" ".join(row) for row in text_split])

    def train(
        self,
        pokedex_list: list[PokedexEntity],
    ):
        pokedex_data_list = [
            self._clean_text(pokedex_entity.data)
            for pokedex_entity in pokedex_list
            if pokedex_entity.data
        ]

        vocab_size = len(self.tokenizer.vocab)
        vocab_size += len(set(" ".join(pokedex_data_list).split()))

        self.tokenizer = self.tokenizer.train_new_from_iterator(
            text_iterator=pokedex_data_list,
            vocab_size=vocab_size,
        )

        return self

    def _chunk_text(
        self,
        text_split: List[str],
    ) -> List[List[str]]:
        text_split_chunked = []
        text_split_padded = text_split + [self.PAD_TOKEN] * self.context_length

        for i in range(0, len(text_split), self.context_length):
            text_split_chunked.append(
                text_split_padded[i : i + self.context_length],
            )

        return text_split_chunked

    def _tokenize_function(
        self,
        batch,
    ) -> dict[str, list]:
        all_names = []
        all_labels = []
        all_chunk_id = []
        all_input_ids = []
        all_input_text = []
        all_original_text = []
        all_attention_masks = []
        all_pokemon_idx = []

        for text, name in zip(batch["text"], batch["name"]):
            text_chunked = self._chunk_text(text.split())
            pokemon_idx = self.name_to_idx[name]

            for i in range(len(text_chunked)):
                all_names.append(name)
                all_chunk_id.append(i + 1)
                all_original_text.append(text)
                all_pokemon_idx.append(pokemon_idx)

                chunk_text = " ".join(text_chunked[i])
                all_input_text.append(chunk_text)

                encoded = self.tokenizer(chunk_text, return_tensors=None)
                all_input_ids.append(encoded["input_ids"])
                all_labels.append(encoded["input_ids"])
                all_attention_masks.append(encoded["attention_mask"])

        return {
            "name": all_names,
            "chunk": all_chunk_id,
            "labels": all_labels,
            "input_ids": all_input_ids,
            "input_text": all_input_text,
            "original_text": all_original_text,
            "attention_mask": all_attention_masks,
            "pokemon_idx": all_pokemon_idx,
        }

    @staticmethod
    def _conditioning_key(name: str) -> str:
        import re

        stem = name.replace(".txt", "")
        stem = re.sub(r"(_frame2)?_flip$", "", stem)
        stem = re.sub(r"_frame2$", "", stem)
        return stem

    @property
    def num_pokemon(self) -> int:
        return len(self.name_to_idx)

    def tokenize(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> DatasetDict:

        filtered = [p for p in pokedex_list if p.data]
        names = [p.name for p in filtered]

        unique_keys = sorted(set(self._conditioning_key(n) for n in names))
        key_to_idx = {key: idx for idx, key in enumerate(unique_keys)}
        self.name_to_idx = {n: key_to_idx[self._conditioning_key(n)] for n in names}

        raw_dataset = Dataset.from_dict(
            {
                "name": names,
                "text": [self._clean_text(p.data) for p in filtered],
            },
        ).cast_column("text", Value("large_string"))

        tokenized_dataset = raw_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        cols = tokenized_dataset.column_names
        if "input_text" in cols:
            tokenized_dataset = tokenized_dataset.cast_column(
                "input_text", Value("large_string")
            )
        if "original_text" in cols:
            tokenized_dataset = tokenized_dataset.cast_column(
                "original_text", Value("large_string")
            )

        return DatasetDict({"train": tokenized_dataset})
