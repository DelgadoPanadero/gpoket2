import random
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from src.application.gld.prof_oak_pc.metadata_adapter import (
    NAME_CHAR_PAD,
    NAME_CHAR_VOCAB,
    NAME_MAX_LEN,
    NUM_EVO_STAGES,
    NUM_GENERATIONS,
    NUM_HAS_EVOLUTION,
    NUM_TYPES1,
    NUM_TYPES2,
    TYPES,
)

_FEATURE_FIELDS = (
    "type1",
    "type2",
    "is_shiny",
    "generation",
    "evolution_stage",
    "has_evolution",
)


class ConditionedGPT2(GPT2LMHeadModel):
    def __init__(
        self,
        config: GPT2Config,
        num_pokemon: int,
        noise_std: float = 0.1,
    ):
        super().__init__(config)
        self.conditioning = nn.Embedding(num_pokemon, config.n_embd)
        self.noise_std = noise_std

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pokemon_idx=None,
        **kwargs,
    ):
        if input_ids is not None and pokemon_idx is not None:
            token_embs = self.transformer.wte(input_ids)
            cond = self.conditioning(pokemon_idx)  # (batch, n_embd)
            if self.training and self.noise_std > 0:
                cond = cond + torch.randn_like(cond) * self.noise_std
            kwargs["inputs_embeds"] = token_embs + cond.unsqueeze(1)
            input_ids = None
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

    def sample_conditioning(self) -> torch.Tensor:
        idx = torch.randint(0, self.conditioning.num_embeddings, (1,))
        with torch.no_grad():
            return self.conditioning(idx)

    def interpolate_conditioning(
        self, idx_a: int, idx_b: int, alpha: float
    ) -> torch.Tensor:
        a = self.conditioning.weight[idx_a].unsqueeze(0)
        b = self.conditioning.weight[idx_b].unsqueeze(0)
        return (1 - alpha) * a + alpha * b

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pokemon_idx=None,
        **kwargs,
    ):
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if pokemon_idx is not None:
            inputs["pokemon_idx"] = pokemon_idx
        return inputs

    def sample_random_conditioning(
        self, device: str = "cpu"
    ) -> torch.Tensor:
        return self.sample_conditioning().to(device)
