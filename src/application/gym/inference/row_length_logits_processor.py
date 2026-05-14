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

    def _pixel_chars(self, token_id: int) -> int:
        return self._tok_len.get(token_id, 0)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq = input_ids[0].tolist()

        current_row, last_pos = -1, -1
        for i, tid in enumerate(seq):
            if tid in self.row_marker_set:
                current_row = self.row_marker_ids.index(tid)
                last_pos = i

        # Before [ROW_00]: let the model emit freely (it should emit [ROW_00])
        if current_row == -1:
            return scores

        chars = sum(self._pixel_chars(tid) for tid in seq[last_pos + 1:])

        if chars >= self.row_width:
            # Row complete: force the next sequential row marker (or EOS at row 63)
            mask = torch.full_like(scores, float("-inf"))
            if current_row < 63:
                mask[0, self.row_marker_ids[current_row + 1]] = 0.0
            else:
                mask[0, self.eos_id] = 0.0
            return mask

        # Row incomplete: block ALL row markers and EOS
        for rid in self.row_marker_ids:
            scores[0, rid] = float("-inf")
        scores[0, self.eos_id] = float("-inf")

        # Also block any pixel token that would overshoot the row quota
        remaining = self.row_width - chars
        for tid, length in self._tok_len.items():
            if length > remaining:
                scores[0, tid] = float("-inf")

        return scores
