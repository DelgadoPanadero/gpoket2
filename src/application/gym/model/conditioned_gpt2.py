import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from src.application.gld.prof_oak_pc.metadata_adapter import (
    NUM_TYPES1,
    NUM_TYPES2,
    NUM_GENERATIONS,
    NUM_EVO_STAGES,
    NUM_HAS_EVOLUTION,
)


class ConditionedGPT2(GPT2LMHeadModel):
    def __init__(
        self,
        config: GPT2Config,
        num_pokemon: int,
        noise_std: float = 0.1,
        row_marker_token_ids: list[int] | None = None,
        num_types1: int = NUM_TYPES1,
        num_types2: int = NUM_TYPES2,
        num_generations: int = NUM_GENERATIONS,
        num_evo_stages: int = NUM_EVO_STAGES,
        token_weights: torch.Tensor | None = None,
    ):
        super().__init__(config)
        self.conditioning = nn.Embedding(num_pokemon, config.n_embd)
        self.noise_std = noise_std

        # Row embedding: 0-63 for sprite rows, 64 = padding (BOS/EOS/pre-row tokens)
        self.row_emb = nn.Embedding(65, config.n_embd, padding_idx=64)
        nn.init.normal_(self.row_emb.weight, std=0.02)
        self.row_emb.weight.data[64].zero_()

        # Metadata conditioning embeddings
        self.type1_emb = nn.Embedding(num_types1, config.n_embd)
        self.type2_emb = nn.Embedding(num_types2, config.n_embd)
        self.is_shiny_emb = nn.Embedding(NUM_HAS_EVOLUTION, config.n_embd)
        self.generation_emb = nn.Embedding(num_generations, config.n_embd)
        self.evo_stage_emb = nn.Embedding(num_evo_stages, config.n_embd)
        self.has_evolution_emb = nn.Embedding(NUM_HAS_EVOLUTION, config.n_embd)

        for emb in (
            self.type1_emb,
            self.type2_emb,
            self.is_shiny_emb,
            self.generation_emb,
            self.evo_stage_emb,
            self.has_evolution_emb,
        ):
            nn.init.normal_(emb.weight, std=0.02)

        # Per-token loss weights — downweights background tokens to focus on color pixels
        self.register_buffer("token_weights", token_weights)

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

    @torch.compiler.disable
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pokemon_idx=None,
        row_ids=None,
        type1=None,
        type2=None,
        is_shiny=None,
        generation=None,
        evolution_stage=None,
        has_evolution=None,
        logits_to_keep: int | torch.Tensor = 0,
        num_items_in_batch=None,
        **kwargs,
    ):
        # Extract labels before passing to parent so we can compute our own loss
        labels = kwargs.pop("labels", None)

        if input_ids is not None and pokemon_idx is not None:
            token_embs = self.transformer.wte(input_ids)

            # Spatial row embedding
            if row_ids is None:
                row_ids = self._ids_to_row_ids(input_ids)
            token_embs = token_embs + self.row_emb(row_ids)

            # Combined conditioning: pokemon identity + metadata
            B, device = token_embs.shape[0], token_embs.device

            def _rand_or_use(val, emb):
                if val is None:
                    val = torch.randint(0, emb.num_embeddings, (B,), device=device)
                return emb(val)

            cond = (
                self.conditioning(pokemon_idx)
                + _rand_or_use(type1, self.type1_emb)
                + _rand_or_use(type2, self.type2_emb)
                + _rand_or_use(is_shiny, self.is_shiny_emb)
                + _rand_or_use(generation, self.generation_emb)
                + _rand_or_use(evolution_stage, self.evo_stage_emb)
                + _rand_or_use(has_evolution, self.has_evolution_emb)
            )

            if self.training and self.noise_std > 0:
                cond = cond + torch.randn_like(cond) * self.noise_std
            token_embs = token_embs + cond.unsqueeze(1)

            kwargs["inputs_embeds"] = token_embs
            input_ids = None

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous().to(outputs.logits.device)
            weights = self.token_weights.to(outputs.logits.device) if self.token_weights is not None else None
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                weight=weights,
                ignore_index=-100,
                reduction="mean",
            )
            outputs.loss = torch.nan_to_num(loss, nan=0.0)

        return outputs

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

        for field in (
            "pokemon_idx",
            "type1",
            "type2",
            "is_shiny",
            "generation",
            "evolution_stage",
            "has_evolution",
        ):
            if field in kwargs:
                inputs[field] = kwargs[field]

        return inputs

    def sample_conditioning(self, idx: int | None = None) -> torch.Tensor:
        if idx is None:
            idx = torch.randint(0, self.conditioning.num_embeddings, (1,)).item()
        with torch.no_grad():
            return self.conditioning(
                torch.tensor([idx], device=self.conditioning.weight.device)
            )

    def sample_random_conditioning(self, device: str = "cpu") -> dict:
        return {
            "pokemon_idx": torch.randint(0, self.conditioning.num_embeddings, (1,), device=device),
            "type1": torch.randint(0, self.type1_emb.num_embeddings, (1,), device=device),
            "type2": torch.randint(0, self.type2_emb.num_embeddings, (1,), device=device),
            "is_shiny": torch.randint(0, self.is_shiny_emb.num_embeddings, (1,), device=device),
            "generation": torch.randint(0, self.generation_emb.num_embeddings, (1,), device=device),
            "evolution_stage": torch.randint(0, self.evo_stage_emb.num_embeddings, (1,), device=device),
            "has_evolution": torch.randint(0, self.has_evolution_emb.num_embeddings, (1,), device=device),
        }
