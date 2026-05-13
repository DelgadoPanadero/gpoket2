import torch
from transformers import LogitsProcessor
from transformers import PreTrainedTokenizerFast


class RowLengthLogitsProcessor(LogitsProcessor):
    """
    Forces each sprite row to contain exactly 64 pixel chars.

    Blocks [ROW_XX+1] and EOS until 64 pixel chars have been generated since
    the last row marker. Once the count reaches 64, forces the next row marker
    (or EOS after row 63).
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

    def _pixel_chars(self, token_id: int) -> int:
        tok = self.id_to_tok.get(token_id, "")
        if tok.startswith("[") and tok.endswith("]"):
            return 0
        return len(tok)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq = input_ids[0].tolist()

        current_row, last_pos = -1, -1
        for i, tid in enumerate(seq):
            if tid in self.row_marker_set:
                current_row = self.row_marker_ids.index(tid)
                last_pos = i

        if current_row == -1:
            return scores

        chars = sum(self._pixel_chars(tid) for tid in seq[last_pos + 1:])

        if chars >= self.row_width:
            mask = torch.full_like(scores, float("-inf"))
            if current_row < 63:
                mask[0, self.row_marker_ids[current_row + 1]] = 0.0
            else:
                mask[0, self.eos_id] = 0.0
            return mask
        else:
            if current_row < 63:
                scores[0, self.row_marker_ids[current_row + 1]] = float("-inf")
            scores[0, self.eos_id] = float("-inf")
            return scores
