import torch
from transformers import LogitsProcessor
from transformers import PreTrainedTokenizerFast


class RowLengthLogitsProcessor(LogitsProcessor):
    """
    Forces each sprite row to contain exactly row_width pixel chars.

    At every step:
    - ALL row markers and EOS are blocked while chars_in_row < row_width
    - Tokens whose char length would exceed the remaining quota are also blocked
      (prevents overshoot, guaranteeing exactly row_width chars per row)
    - Once chars_in_row >= row_width, forces the next sequential row marker
      (or EOS after row 63) and blocks everything else

    State is tracked incrementally (O(1) per step instead of O(seq_len)),
    and overshoot masks are precomputed as tensors to avoid Python loops at
    generation time.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        row_marker_ids: list[int],
        row_width: int = 64,
    ):
        self.id_to_tok = {v: k for k, v in tokenizer.get_vocab().items()}
        self.row_marker_ids = row_marker_ids
        self.row_marker_set = set(row_marker_ids)
        self.eos_id = tokenizer.eos_token_id
        self.row_width = row_width

        # Precompute pixel length for every token id (0 for special tokens)
        self._tok_len: dict[int, int] = {}
        for tid, tok in self.id_to_tok.items():
            if tok.startswith("[") and tok.endswith("]"):
                self._tok_len[tid] = 0
            else:
                self._tok_len[tid] = len(tok)

        vocab_size = len(tokenizer.get_vocab())

        # Precompute row-marker block mask (tensor, applied in one op).
        # Also blocks zero-length special tokens (PAD, BOS, UNK) that are not
        # row markers — they don't consume chars_in_row quota but still occupy
        # a token slot, which would push the row over the expected token count.
        row_marker_tensor = torch.tensor(row_marker_ids, dtype=torch.long)
        self._row_block_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self._row_block_mask[row_marker_tensor] = True
        self._row_block_mask[self.eos_id] = True
        for tid, length in self._tok_len.items():
            if length == 0:
                self._row_block_mask[tid] = True

        # Precompute overshoot masks: _overshoot_masks[r] blocks tokens that
        # would use more than r chars, for r in 1..row_width.
        # Index 0 is unused; remaining=0 never occurs (row would be complete).
        self._overshoot_masks: list[torch.Tensor] = [
            torch.zeros(vocab_size, dtype=torch.bool),
        ]
        for remaining in range(1, row_width + 1):
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            for tid, length in self._tok_len.items():
                if length > remaining:
                    mask[tid] = True
            self._overshoot_masks.append(mask)

        self._reset_state()

    def _reset_state(self) -> None:
        self._current_row: int = -1
        self._chars_in_row: int = 0
        self._state_seq_len: int = 0

    def _pixel_chars(self, token_id: int) -> int:
        return self._tok_len.get(token_id, 0)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        seq_len = input_ids.shape[1]

        # New generation detected (sequence shorter than our state) — reset
        if seq_len <= self._state_seq_len:
            self._reset_state()

        # Incrementally process only newly added tokens
        new_tokens = input_ids[0, self._state_seq_len :].tolist()
        for tid in new_tokens:
            if tid in self.row_marker_set:
                self._current_row = self.row_marker_ids.index(tid)
                self._chars_in_row = 0
            else:
                self._chars_in_row += self._tok_len.get(tid, 0)
        self._state_seq_len = seq_len

        # Before [ROW_00]: let the model emit freely
        if self._current_row == -1:
            return scores

        device = scores.device

        if self._chars_in_row >= self.row_width:
            # Row complete: force next row marker (or EOS at row 63)
            mask = torch.full_like(scores, float("-inf"))
            if self._current_row < 63:
                mask[0, self.row_marker_ids[self._current_row + 1]] = 0.0
            else:
                mask[0, self.eos_id] = 0.0
            return mask

        # Row incomplete: block row markers, EOS, and overshoot tokens
        scores[0, self._row_block_mask.to(device)] = float("-inf")
        remaining = self.row_width - self._chars_in_row
        remaining = min(remaining, self.row_width)
        scores[0, self._overshoot_masks[remaining].to(device)] = float("-inf")

        return scores
