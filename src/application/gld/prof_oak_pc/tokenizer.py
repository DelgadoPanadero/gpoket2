from typing import List

from datasets import Value
from datasets import Dataset
from datasets import Sequence
from datasets import DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast  # type: ignore
from tokenizers.pre_tokenizers import WhitespaceSplit

from src.domain.slv.pokedex import PokedexEntity


class Pokenizer:
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    BOL_TOKEN = "00"
    BCK_TOKEN = "~"

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

        _tokenizer = Tokenizer(WordLevel())  # type: ignore
        _tokenizer.pre_tokenizer = WhitespaceSplit()  # type: ignore

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=_tokenizer,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            pad_token=self.EOS_TOKEN,  # GPT-2 no necesita padding; para el collator lo igualamos a eos
        )

    # def to_dict(self) -> dict:
    #    return json.loads(self._tokenizer.to_str())

    def _clean_text(
        self,
        text: str,
    ) -> str:
        text_split = text.split("\n")
        # text_split = [[self.BOL_TOKEN] + row.split()[1:] for row in text_split]
        text_split = [["%02d" % i]+r.split()[1:] for i,r in enumerate(text_split)]
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
        vocab_size += len(set(" ".join(pokedex_data_list).split()))  # añadir nuevas palabras

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
        step = self.chunk_step_rows * self.row_length
        text_split_padded = text_split + [self.EOS_TOKEN] * self.context_length

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
        all_labels = []
        all_chunk_id = []
        all_input_ids = []
        all_input_text = []
        all_original_text = []
        all_attention_masks = []
        for text, name in zip(batch["text"], batch["name"]):
            text_chunked = self._chunk_text(text.split())

            for i in range(len(text_chunked)):
                all_names.append(name)

                all_chunk_id.append(i + 1)

                all_original_text.append(text)

                all_input_text.append(
                    " ".join(text_chunked[i]),
                )

                all_input_ids.append(
                    self.tokenizer(
                        " ".join(text_chunked[i]),
                        return_tensors=None,
                    )["input_ids"]
                )

                all_labels.append(
                    self.tokenizer(
                        " ".join(text_chunked[i]),
                        return_tensors=None,
                    )["input_ids"]
                )
                all_attention_masks.append(
                    self.tokenizer(
                        " ".join(text_chunked[i]),
                        return_tensors=None,
                    )["attention_mask"],
                )

        return {
            "name": all_names,
            "chunk": all_chunk_id,
            "labels": all_labels,
            "input_ids": all_input_ids,
            "input_text": all_input_text,
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
        ).cast_column("text", Value("large_string"))

        tokenized_dataset = (
            raw_dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=["text"],
            )
            .cast_column("input_text", Value("large_string"))
            .cast_column("original_text", Value("large_string"))
            .cast_column("labels", Sequence(Value("int64")))
            .cast_column("input_ids", Sequence(Value("int64")))
        )

        return DatasetDict({"train": tokenized_dataset})
