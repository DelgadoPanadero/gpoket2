import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class ConditionedGPT2(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config, num_pokemon: int):
        super().__init__(config)
        self.conditioning = nn.Embedding(num_pokemon, config.n_embd)

    def forward(
        self,
        input_ids=None,
        pokemon_idx=None,
        **kwargs,
    ):
        if input_ids is not None and pokemon_idx is not None:
            token_embs = self.transformer.wte(input_ids)
            cond = self.conditioning(pokemon_idx)  # (batch, n_embd)
            kwargs["inputs_embeds"] = token_embs + cond.unsqueeze(1)
            input_ids = None
        return super().forward(input_ids=input_ids, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pokemon_idx=None,
        **kwargs,
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            **kwargs,
        )
        if pokemon_idx is not None:
            inputs["pokemon_idx"] = pokemon_idx
        return inputs

    def sample_conditioning(
        self,
        device: str = "cpu",
    ) -> torch.Tensor:
        std = self.conditioning.weight.std().item()
        return torch.randn(1, self.config.n_embd, device=device) * std

    def interpolate_conditioning(
        self,
        idx_a: int,
        idx_b: int,
        alpha: float = 0.5,
        device: str = "cpu",
    ) -> torch.Tensor:
        emb_a = self.conditioning.weight[idx_a]
        emb_b = self.conditioning.weight[idx_b]
        return ((1 - alpha) * emb_a + alpha * emb_b).unsqueeze(0).to(device)
