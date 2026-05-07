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
        noise_std: float = 0.1,
        name_dropout: float = 0.5,
    ):
        super().__init__(config)
        n = config.n_embd
        self.type1_emb = nn.Embedding(NUM_TYPES1, n)
        self.type2_emb = nn.Embedding(NUM_TYPES2, n)
        self.shiny_emb = nn.Embedding(2, n)
        self.generation_emb = nn.Embedding(NUM_GENERATIONS, n)
        self.evolution_stage_emb = nn.Embedding(NUM_EVO_STAGES, n)
        self.has_evolution_emb = nn.Embedding(NUM_HAS_EVOLUTION, n)
        self.name_char_emb = nn.Embedding(
            NAME_CHAR_VOCAB, n, padding_idx=NAME_CHAR_PAD
        )
        self.noise_std = noise_std
        self.name_dropout = name_dropout

    def _encode_name(self, name_chars: torch.Tensor) -> torch.Tensor:
        emb = self.name_char_emb(name_chars)  # (batch, max_len, n)
        mask = (
            (name_chars != NAME_CHAR_PAD).float().unsqueeze(-1)
        )  # (batch, max_len, 1)
        return (emb * mask).sum(1) / mask.sum(1).clamp(min=1)  # (batch, n)

    def _conditioning(
        self,
        type1: torch.Tensor,
        type2: torch.Tensor,
        is_shiny: torch.Tensor,
        generation: torch.Tensor,
        evolution_stage: torch.Tensor,
        has_evolution: torch.Tensor,
        name_chars: torch.Tensor,
    ) -> torch.Tensor:
        name_vec = self._encode_name(name_chars)
        if self.training and self.name_dropout > 0:
            drop_mask = (
                torch.rand(name_vec.shape[0], device=name_vec.device)
                > self.name_dropout
            ).float()
            name_vec = name_vec * drop_mask.unsqueeze(1)

        cond = (
            self.type1_emb(type1)
            + self.type2_emb(type2)
            + self.shiny_emb(is_shiny)
            + self.generation_emb(generation)
            + self.evolution_stage_emb(evolution_stage)
            + self.has_evolution_emb(has_evolution)
            + name_vec
        )
        if self.training and self.noise_std > 0:
            cond = cond + torch.randn_like(cond) * self.noise_std
        return cond  # (batch, n)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        type1=None,
        type2=None,
        is_shiny=None,
        generation=None,
        evolution_stage=None,
        has_evolution=None,
        name_chars=None,
        **kwargs,
    ):
        feature_vals = (
            type1,
            type2,
            is_shiny,
            generation,
            evolution_stage,
            has_evolution,
            name_chars,
        )
        if input_ids is not None and all(v is not None for v in feature_vals):
            token_embs = self.transformer.wte(input_ids)
            cond = self._conditioning(
                type1,
                type2,
                is_shiny,
                generation,
                evolution_stage,
                has_evolution,
                name_chars,
            )
            kwargs["inputs_embeds"] = token_embs + cond.unsqueeze(1)
            input_ids = None
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        type1=None,
        type2=None,
        is_shiny=None,
        generation=None,
        evolution_stage=None,
        has_evolution=None,
        name_chars=None,
        **kwargs,
    ):
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        feature_vals = (
            type1,
            type2,
            is_shiny,
            generation,
            evolution_stage,
            has_evolution,
            name_chars,
        )
        if all(v is not None for v in feature_vals):
            inputs["type1"] = type1
            inputs["type2"] = type2
            inputs["is_shiny"] = is_shiny
            inputs["generation"] = generation
            inputs["evolution_stage"] = evolution_stage
            inputs["has_evolution"] = has_evolution
            inputs["name_chars"] = name_chars
        return inputs

    def sample_random_conditioning(
        self, device: str = "cpu"
    ) -> dict[str, torch.Tensor]:
        length = random.randint(4, 10)
        chars = [random.randint(ord("a"), ord("z")) for _ in range(length)]
        chars += [NAME_CHAR_PAD] * (NAME_MAX_LEN - length)
        return {
            "type1": torch.randint(
                0,
                len(TYPES),
                (1,),
                device=device,
            ),
            "type2": torch.randint(
                0,
                NUM_TYPES2,
                (1,),
                device=device,
            ),
            "is_shiny": torch.randint(
                0,
                2,
                (1,),
                device=device,
            ),
            "generation": torch.randint(
                0,
                NUM_GENERATIONS,
                (1,),
                device=device,
            ),
            "evolution_stage": torch.randint(
                0,
                NUM_EVO_STAGES,
                (1,),
                device=device,
            ),
            "has_evolution": torch.randint(
                0,
                NUM_HAS_EVOLUTION,
                (1,),
                device=device,
            ),
            "name_chars": torch.tensor(
                [chars],
                dtype=torch.long,
                device=device,
            ),
        }
