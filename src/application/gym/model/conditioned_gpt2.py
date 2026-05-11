import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class ConditionedGPT2(GPT2LMHeadModel):
    def __init__(
        self,
        config: GPT2Config,
        num_pokemon: int,
        noise_std: float = 0.1,
        row_marker_token_ids: list[int] | None = None,
    ):
        super().__init__(config)
        self.conditioning = nn.Embedding(num_pokemon, config.n_embd)
        self.noise_std = noise_std

        # Row embedding: 0-63 for sprite rows, 64 = padding (BOS/EOS/pre-row tokens)
        self.row_emb = nn.Embedding(65, config.n_embd, padding_idx=64)
        nn.init.normal_(self.row_emb.weight, std=0.02)
        self.row_emb.weight.data[64].zero_()

        # Store row marker token ids as a buffer so they're saved with the model
        _ids = row_marker_token_ids or [0] * 64
        self.register_buffer(
            "row_marker_ids",
            torch.tensor(_ids, dtype=torch.long),
        )

    def _ids_to_row_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the row index (0-63) for each token in input_ids.
        Tokens before the first [ROW_XX] marker get index 64 (padding).
        Uses forward-fill: each token inherits the row of the most recent marker.
        """
        B, T = input_ids.shape
        row_ids = input_ids.new_full((B, T), 64)

        # Mark positions that are row-marker tokens
        for row_idx in range(64):
            tok_id = self.row_marker_ids[row_idx]
            row_ids[input_ids == tok_id] = row_idx

        # Forward-fill: propagate row index to subsequent non-marker positions
        for t in range(1, T):
            inherit = row_ids[:, t] == 64
            row_ids[:, t] = torch.where(inherit, row_ids[:, t - 1], row_ids[:, t])

        return row_ids

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pokemon_idx=None,
        row_ids=None,
        **kwargs,
    ):
        if input_ids is not None and pokemon_idx is not None:
            token_embs = self.transformer.wte(input_ids)

            # Spatial row embedding
            if row_ids is None:
                row_ids = self._ids_to_row_ids(input_ids)
            token_embs = token_embs + self.row_emb(row_ids)

            # Pokemon conditioning
            cond = self.conditioning(pokemon_idx)  # (batch, n_embd)
            if self.training and self.noise_std > 0:
                cond = cond + torch.randn_like(cond) * self.noise_std
            token_embs = token_embs + cond.unsqueeze(1)

            kwargs["inputs_embeds"] = token_embs
            input_ids = None

        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Compute row_ids from the full sequence before the parent trims it for KV cache
        row_ids_full = self._ids_to_row_ids(input_ids)

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )

        # Trim row_ids the same way the parent trims input_ids
        if past_key_values is not None:
            inputs["row_ids"] = row_ids_full[:, -1:]
        else:
            inputs["row_ids"] = row_ids_full

        if "pokemon_idx" in kwargs:
            inputs["pokemon_idx"] = kwargs["pokemon_idx"]

        return inputs

    def sample_conditioning(self, idx: int | None = None) -> torch.Tensor:
        if idx is None:
            idx = torch.randint(0, self.conditioning.num_embeddings, (1,)).item()
        with torch.no_grad():
            return self.conditioning(
                torch.tensor([idx], device=self.conditioning.weight.device)
            )

    def sample_random_conditioning(self, device: str = "cpu") -> dict:
        idx = torch.randint(0, self.conditioning.num_embeddings, (1,)).item()
        return {"pokemon_idx": torch.tensor([idx], device=device)}
