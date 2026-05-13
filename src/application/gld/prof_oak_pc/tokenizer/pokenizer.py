import re
from typing import List

from datasets import Value
from datasets import Dataset
from datasets import DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from src.domain.slv.pokedex import PokedexEntity

_ROW_TOKENS = [f"[ROW_{i:02d}]" for i in range(64)]


class Pokenizer:
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"

    _SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + _ROW_TOKENS

    def __init__(
        self,
        row_length: int = 64,
        col_length: int = 64,
        context_length: int = 1024,
    ):
        self.row_length = row_length
        self.col_length = col_length
        self.context_length = context_length
        self._row_tid_to_row: dict[int, int] = {}

        _bpe = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
        _bpe.pre_tokenizer = WhitespaceSplit()

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=_bpe,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            pad_token=self.PAD_TOKEN,
            unk_token=self.UNK_TOKEN,
        )

    def _clean_text(self, text: str) -> str:
        lines = [line for line in text.split("\n") if line.strip()]
        parts: list[str] = []
        for pos, line in enumerate(lines):
            parts.append(f"[ROW_{pos:02d}]")
            # Split each row into 8-char chunks so BPE merges stay within
            # chunk boundaries. The model must emit exactly 8 chunks per row,
            # making the 64-pixel constraint learnable without explicit counting.
            pixels = "".join(line.split())
            for i in range(0, self.row_length, 8):
                parts.append(pixels[i : i + 8] or "~" * 8)
        return " ".join(parts)

    def train(self, pokedex_list: list[PokedexEntity]):
        corpus = [
            self._clean_text(e.data)
            for e in pokedex_list
            if e.data
        ]

        _bpe = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
        _bpe.pre_tokenizer = WhitespaceSplit()

        trainer = BpeTrainer(
            vocab_size=3000,
            special_tokens=self._SPECIAL_TOKENS,
            min_frequency=2,
        )
        _bpe.train_from_iterator(corpus, trainer=trainer)

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=_bpe,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            pad_token=self.PAD_TOKEN,
            unk_token=self.UNK_TOKEN,
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": _ROW_TOKENS}
        )
        self._build_row_tid_map()
        return self

    def _build_row_tid_map(self):
        self._row_tid_to_row = {}
        for i, tok in enumerate(_ROW_TOKENS):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid != self.tokenizer.unk_token_id:
                self._row_tid_to_row[tid] = i

    def _compute_row_ids(self, token_ids: list[int]) -> list[int]:
        row_ids: list[int] = []
        current_row = 64  # 64 = padding value (before first row marker)
        for tid in token_ids:
            if tid in self._row_tid_to_row:
                current_row = self._row_tid_to_row[tid]
            row_ids.append(current_row)
        return row_ids

    def _tokenize_function(self, batch) -> dict[str, list]:
        all_names: list = []
        all_labels: list = []
        all_chunk_id: list = []
        all_input_ids: list = []
        all_input_text: list = []
        all_original_text: list = []
        all_attention_masks: list = []
        all_pokemon_idx: list = []
        all_row_ids: list = []

        pad_id = self.tokenizer.pad_token_id

        for text, name in zip(batch["text"], batch["name"]):
            pokemon_idx = self.name_to_idx[name]

            encoded = self.tokenizer(text, truncation=False, return_tensors=None)
            token_ids: list[int] = (
                [self.tokenizer.bos_token_id]
                + encoded["input_ids"]
                + [self.tokenizer.eos_token_id]
            )
            row_ids = [64] + self._compute_row_ids(encoded["input_ids"]) + [64]

            # Chunk — most sprites fit in one chunk at context_length=1024
            n = max(1, len(token_ids))
            for chunk_num, i in enumerate(range(0, n, self.context_length)):
                chunk_ids = token_ids[i : i + self.context_length]
                chunk_row_ids = row_ids[i : i + self.context_length]

                pad_len = self.context_length - len(chunk_ids)
                padded_ids = chunk_ids + [pad_id] * pad_len
                padded_row_ids = chunk_row_ids + [64] * pad_len
                attn_mask = [1] * len(chunk_ids) + [0] * pad_len

                all_names.append(name)
                all_chunk_id.append(chunk_num + 1)
                all_original_text.append(text)
                all_pokemon_idx.append(pokemon_idx)
                all_input_text.append(text[: 512])  # truncate for storage only
                all_input_ids.append(padded_ids)
                all_labels.append(padded_ids)
                all_attention_masks.append(attn_mask)
                all_row_ids.append(padded_row_ids)

        return {
            "name": all_names,
            "chunk": all_chunk_id,
            "labels": all_labels,
            "input_ids": all_input_ids,
            "input_text": all_input_text,
            "original_text": all_original_text,
            "attention_mask": all_attention_masks,
            "pokemon_idx": all_pokemon_idx,
            "row_ids": all_row_ids,
        }

    @staticmethod
    def _conditioning_key(name: str) -> str:
        stem = name.replace(".txt", "")
        stem = re.sub(r"(_frame2)?_flip$", "", stem)
        stem = re.sub(r"_frame2$", "", stem)
        return stem

    @property
    def num_pokemon(self) -> int:
        return len(self.name_to_idx)

    def tokenize(self, pokedex_list: list[PokedexEntity]) -> DatasetDict:
        filtered = [p for p in pokedex_list if p.data]
        names = [p.name for p in filtered]

        unique_keys = sorted(set(self._conditioning_key(n) for n in names))
        key_to_idx = {key: idx for idx, key in enumerate(unique_keys)}
        self.name_to_idx = {n: key_to_idx[self._conditioning_key(n)] for n in names}

        raw_dataset = Dataset.from_dict(
            {
                "name": names,
                "text": [self._clean_text(p.data) for p in filtered],
            }
        ).cast_column("text", Value("large_string"))

        tokenized_dataset = raw_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        for col in ("input_text", "original_text"):
            if col in tokenized_dataset.column_names:
                tokenized_dataset = tokenized_dataset.cast_column(
                    col, Value("large_string")
                )

        return DatasetDict({"train": tokenized_dataset})

    def row_marker_token_ids(self) -> list[int]:
        """Returns the 64 token ids corresponding to [ROW_00]..[ROW_63], in order."""
        return [
            self.tokenizer.convert_tokens_to_ids(tok) for tok in _ROW_TOKENS
        ]
