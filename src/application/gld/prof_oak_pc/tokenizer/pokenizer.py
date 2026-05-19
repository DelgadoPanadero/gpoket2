import re
from typing import List

from datasets import Value
from datasets import Dataset
from datasets import DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from src.domain.slv.pokedex import PokedexEntity

_ROW_TOKENS = [f"[ROW_{i:02d}]" for i in range(64)]

_COL_PAD = 64  # padding value for col_ids (row markers and non-pixel tokens)

# Fixed positional patterns for a valid 64×64 sprite:
# structure is always ([ROW_XX] + 63 pixels) × 64 rows = 4096 tokens
_ROW_IDS_PATTERN: list[int] = []
_COL_IDS_PATTERN: list[int] = []
for _r in range(64):
    _ROW_IDS_PATTERN.append(_r)  # row marker → its own row
    _ROW_IDS_PATTERN.extend([_r] * 63)  # 63 pixels → same row
    _COL_IDS_PATTERN.append(_COL_PAD)  # row marker → padding
    _COL_IDS_PATTERN.extend(range(63))  # pixels → col 0..62


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
        context_length: int = 4096,
    ):
        self.row_length = row_length
        self.col_length = col_length
        self.context_length = context_length
        self._row_tid_to_row: dict[int, int] = {}

        _wl = Tokenizer(WordLevel(unk_token=self.UNK_TOKEN))
        _wl.pre_tokenizer = WhitespaceSplit()

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=_wl,
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
            parts.extend(
                line.split()[1 : self.row_length],
            )  # drop first pixel → 63 pixels/row
        return " ".join(parts)

    def train(self, pokedex_list: list[PokedexEntity]):
        corpus = [self._clean_text(e.data) for e in pokedex_list if e.data]

        all_tokens: set[str] = set()
        for text in corpus:
            all_tokens.update(text.split())

        vocab: dict[str, int] = {
            tok: i for i, tok in enumerate(self._SPECIAL_TOKENS)
        }
        for tok in sorted(all_tokens - set(self._SPECIAL_TOKENS)):
            vocab[tok] = len(vocab)

        _wl = Tokenizer(WordLevel(vocab=vocab, unk_token=self.UNK_TOKEN))
        _wl.pre_tokenizer = WhitespaceSplit()

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=_wl,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            pad_token=self.PAD_TOKEN,
            unk_token=self.UNK_TOKEN,
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
        all_col_ids: list = []

        pad_id = self.tokenizer.pad_token_id

        for text, name in zip(batch["text"], batch["name"]):
            pokemon_idx = self.name_to_idx[name]

            # WordLevel: 1 word = 1 token — encode entire text in one call.
            token_ids_raw: list[int] = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            # row_ids and col_ids follow a fixed pattern for any valid 64×64 sprite.
            n = len(token_ids_raw)
            token_ids = token_ids_raw
            row_ids = _ROW_IDS_PATTERN[:n]
            col_ids = _COL_IDS_PATTERN[:n]

            n = max(1, len(token_ids))
            for chunk_num, i in enumerate(range(0, n, self.context_length)):
                chunk_ids = token_ids[i : i + self.context_length]
                chunk_row_ids = row_ids[i : i + self.context_length]
                chunk_col_ids = col_ids[i : i + self.context_length]

                pad_len = self.context_length - len(chunk_ids)
                padded_ids = chunk_ids + [pad_id] * pad_len
                padded_row_ids = chunk_row_ids + [64] * pad_len
                padded_col_ids = chunk_col_ids + [_COL_PAD] * pad_len
                attn_mask = [1] * len(chunk_ids) + [0] * pad_len

                all_names.append(name)
                all_chunk_id.append(chunk_num + 1)
                all_original_text.append(text)
                all_pokemon_idx.append(pokemon_idx)
                all_input_text.append(text[:512])  # truncate for storage only
                all_input_ids.append(padded_ids)
                all_labels.append(padded_ids)
                all_attention_masks.append(attn_mask)
                all_row_ids.append(padded_row_ids)
                all_col_ids.append(padded_col_ids)

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
            "col_ids": all_col_ids,
        }

    @staticmethod
    def _conditioning_key(name: str) -> str:
        stem = name.replace(".txt", "")
        stem = re.sub(r"(_frame2)?_flip$", "", stem)
        stem = re.sub(r"_frame2$", "", stem)
        stem = re.sub(r"_(rg|rb|gb|cyc1|cyc2)$", "", stem)
        return stem

    @property
    def num_pokemon(self) -> int:
        return len(self.name_to_idx)

    def tokenize(self, pokedex_list: list[PokedexEntity]) -> DatasetDict:
        filtered = [p for p in pokedex_list if p.data]
        names = [p.name for p in filtered]

        unique_keys = sorted(set(self._conditioning_key(n) for n in names))
        key_to_idx = {key: idx for idx, key in enumerate(unique_keys)}
        self.name_to_idx = {
            n: key_to_idx[self._conditioning_key(n)] for n in names
        }

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

        for col in ("input_text", "original_text"):
            if col in tokenized_dataset.column_names:
                tokenized_dataset = tokenized_dataset.cast_column(
                    col,
                    Value("large_string"),
                )

        return DatasetDict({"train": tokenized_dataset})

    def row_marker_token_ids(self) -> list[int]:
        """Returns the 64 token ids corresponding to [ROW_00]..[ROW_63], in order."""
        return [
            self.tokenizer.convert_tokens_to_ids(tok) for tok in _ROW_TOKENS
        ]
