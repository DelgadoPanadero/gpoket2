import json
from typing import List

from datasets import Dataset
from datasets import DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import WhitespaceSplit

from src.domain.slv.pokedex import PokedexEntity


class Pokenizer:

    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"
    BOL_TOKEN = "00"
    BCK_TOKEN = "~"

    special_words = [
        BOL_TOKEN,
        BCK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
    ]

    def __init__(
        self,
        row_length: int = 64,
        col_length: int = 64,
        chunk_step_rows: int = 1,
        context_length: int = 4096,
    ):
        self.row_length = row_length
        self.col_length = col_length
        self.context_length = context_length
        self.chunk_step_rows = chunk_step_rows

        self._tokenizer = Tokenizer(WordLevel())
        self._tokenizer.pre_tokenizer = WhitespaceSplit()

    def to_dict(self) -> dict:
        return json.loads(self._tokenizer.to_str())

    def _clean_text(
        self,
        text: str,
    ) -> str:

        text_split = text.split("\n")
        text_split = [["00"] + r.split() for i, r in enumerate(text_split)]
        # text_split = [["%02d" % i]+r.split() for i,r in enumerate(text_split)]
        text_split = [row[0 : self.row_length] for row in text_split]
        text_split[+0][+0] = self.BOS_TOKEN
        text_split[-1][-1] = self.EOS_TOKEN

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

        pokedex_data_vocab = [
            word
            for word in set(
                " ".join([pkd for pkd in pokedex_data_list]).split(" ")
            )
            if word not in self.special_words
        ]

        pokedex_data_tokens = {
            word: token
            for token, word in enumerate(
                self.special_words + pokedex_data_vocab
            )
        }

        self._tokenizer = Tokenizer(WordLevel(vocab=pokedex_data_tokens))
        self._tokenizer.pre_tokenizer = WhitespaceSplit()

        return self

    def _chunk_text(
        self,
        text_split: List[str],
    ) -> List[List[str]]:

        text_split_chunked = []
        step = self.chunk_step_rows * self.row_length
        text_split_padded = text_split + [self.PAD_TOKEN] * self.context_length

        for i in range(0, len(text_split) - self.context_length + 1, step):
            text_split_chunked.append(
                text_split_padded[i : i + self.context_length],
            )

        return text_split_chunked

    def _tokenize_function(
        self,
        batch,
    ) -> dict[str, list]:

        all_names = []
        all_chunk_id = []
        all_input_ids = []
        all_inputs_text = []
        all_original_text = []
        all_attention_masks = []
        for text, name in zip(batch["text"], batch["name"]):

            text_chunked = self._chunk_text(text.split())

            for i in range(len(text_chunked)):

                all_names.append(name)

                all_chunk_id.append(i + 1)

                all_original_text.append(batch["text"])

                all_inputs_text.append(" ".join(text_chunked[i]))

                all_input_ids.append(
                    self._tokenizer.encode(" ".join(text_chunked[i])).ids,
                )

                all_attention_masks.append(
                    self._tokenizer.encode(
                        " ".join(text_chunked[i])
                    ).attention_mask,
                )

        return {
            "name": all_names,
            "chunk": all_chunk_id,
            "input_ids": all_input_ids,
            "inputs_text": all_inputs_text,
            "original_text": all_original_text,
            "attention_mask": all_attention_masks,
        }

    def tokenize(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> DatasetDict:

        raw_dataset = Dataset.from_dict(
            {
                "name": [
                    pokedex_entity.name
                    for pokedex_entity in pokedex_list
                    if pokedex_entity.data
                ],
                "text": [
                    self._clean_text(pokedex_entity.data)
                    for pokedex_entity in pokedex_list
                    if pokedex_entity.data
                ],
            }
        )

        tokenized_dataset = raw_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        return DatasetDict({"train": tokenized_dataset})
