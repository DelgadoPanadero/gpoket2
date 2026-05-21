import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor

NUM_GENERATIONS = 10
NUM_TYPES1 = 19  # 18 tipos + UNK
NUM_TYPES2 = 20  # 18 tipos + NONE + UNK
NUM_EVO_STAGES = 4  # 3 etapas + 1
NUM_HAS_EVOLUTION = 2
NUM_COLOR_SHIFTS = 6  # 0 = no shift, 1-5 = ColorShift permutations

_TYPES = [
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
]
_TYPE1_UNK = 18  # unknown primary type
_TYPE2_NONE = 18  # no secondary type
_TYPE2_UNK = 19  # unknown secondary type


def _resolve_type(
    val: "str | int | None", default_none_idx: int
) -> "int | None":
    if val is None:
        return None
    if isinstance(val, str):
        return (
            _TYPES.index(val.lower())
            if val.lower() in _TYPES
            else default_none_idx
        )
    return int(val)


class ConditionedGPT2(GPT2LMHeadModel):
    def __init__(
        self,
        config: GPT2Config,
        num_pokemon: int | None = None,
        noise_std: float = 0.1,
        row_marker_token_ids: list[int] | None = None,
        num_types1: int = NUM_TYPES1,
        num_types2: int = NUM_TYPES2,
        num_generations: int = NUM_GENERATIONS,
        num_evo_stages: int = NUM_EVO_STAGES,
        token_weights: torch.Tensor | None = None,
        num_color_shifts: int = NUM_COLOR_SHIFTS,
    ):
        num_pokemon = num_pokemon or getattr(config, "num_pokemon", None)
        if num_pokemon is None:
            raise ValueError(
                "num_pokemon must be provided or present in config"
            )
        config.num_pokemon = num_pokemon
        super().__init__(config)
        self.conditioning = nn.Embedding(num_pokemon, config.n_embd)
        nn.init.normal_(self.conditioning.weight, std=0.02)
        self.noise_std = noise_std

        # Row embedding: 0-63 for sprite rows, 64 = padding (BOS/EOS/pre-row tokens)
        self.row_emb = nn.Embedding(65, config.n_embd, padding_idx=64)
        nn.init.normal_(self.row_emb.weight, std=0.02)
        self.row_emb.weight.data[64].zero_()

        # Column embedding: 0-63 for pixel position within a row, 64 = padding
        self.col_emb = nn.Embedding(65, config.n_embd, padding_idx=64)
        nn.init.normal_(self.col_emb.weight, std=0.02)
        self.col_emb.weight.data[64].zero_()

        # Metadata conditioning embeddings
        self.type1_emb = nn.Embedding(num_types1, config.n_embd)
        self.type2_emb = nn.Embedding(num_types2, config.n_embd)
        self.is_shiny_emb = nn.Embedding(NUM_HAS_EVOLUTION, config.n_embd)
        self.generation_emb = nn.Embedding(num_generations, config.n_embd)
        self.evo_stage_emb = nn.Embedding(num_evo_stages, config.n_embd)
        self.has_evolution_emb = nn.Embedding(NUM_HAS_EVOLUTION, config.n_embd)
        self.color_shift_emb = nn.Embedding(num_color_shifts, config.n_embd)

        for emb in (
            self.type1_emb,
            self.type2_emb,
            self.is_shiny_emb,
            self.generation_emb,
            self.evo_stage_emb,
            self.has_evolution_emb,
            self.color_shift_emb,
        ):
            nn.init.normal_(emb.weight, std=0.02)

        # Per-token loss weights — downweights background tokens to focus on color pixels.
        # Never None so PyTorch always includes it in state_dict and from_pretrained
        # can load it from checkpoint without an "unexpected key" warning.
        if token_weights is None:
            token_weights = torch.ones(config.vocab_size)
        self.register_buffer("token_weights", token_weights)

        # Store row marker token ids as a buffer so they're saved with the model
        _ids = row_marker_token_ids or [0] * 64
        self.register_buffer(
            "row_marker_ids",
            torch.tensor(_ids, dtype=torch.long),
        )

    def _ids_to_row_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        row_ids = input_ids.new_full((B, T), 64)
        for row_idx in range(64):
            row_ids[input_ids == self.row_marker_ids[row_idx]] = row_idx

        # Vectorized forward-fill via cummax: for each position find the last marker seen
        is_assigned = row_ids < 64
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        last_marker_t, _ = torch.where(
            is_assigned,
            t_idx,
            torch.zeros_like(t_idx),
        ).cummax(dim=1)
        row_ids_filled = torch.gather(row_ids, 1, last_marker_t)

        in_row = is_assigned.long().cumsum(dim=1) >= 1
        return torch.where(
            in_row,
            row_ids_filled,
            input_ids.new_full((B, T), 64),
        )

    def _ids_to_col_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        is_marker = torch.isin(input_ids, self.row_marker_ids.to(device))
        is_pixel = ~is_marker
        in_row = is_marker.long().cumsum(dim=1) >= 1

        # Cumulative pixel count (inclusive) and the baseline at the last marker
        pixel_cumsum = is_pixel.long().cumsum(dim=1)
        marker_base = torch.where(
            is_marker,
            pixel_cumsum,
            torch.zeros_like(pixel_cumsum),
        )
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        last_marker_t, _ = torch.where(
            is_marker,
            t_idx,
            torch.zeros_like(t_idx),
        ).cummax(dim=1)
        last_marker_base = torch.gather(marker_base, 1, last_marker_t)

        # 0-indexed column position within the current row
        col_pos = pixel_cumsum - last_marker_base - 1
        return torch.where(
            is_pixel & in_row & (col_pos < 64),
            col_pos.clamp(min=0),
            input_ids.new_full((B, T), 64),
        )

    @torch.compiler.disable
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pokemon_idx=None,
        pokemon_cond=None,
        row_ids=None,
        col_ids=None,
        type1=None,
        type2=None,
        is_shiny=None,
        generation=None,
        evolution_stage=None,
        has_evolution=None,
        color_shift=None,
        logits_to_keep: int | torch.Tensor = 0,
        num_items_in_batch=None,
        **kwargs,
    ):
        # Extract labels before passing to parent so we can compute our own loss
        labels = kwargs.pop("labels", None)

        if input_ids is not None and (
            pokemon_idx is not None or pokemon_cond is not None
        ):
            token_embs = self.transformer.wte(input_ids)

            # Spatial 2-D embeddings: row (which row) + col (which pixel position)
            if row_ids is None:
                row_ids = self._ids_to_row_ids(input_ids)
            if col_ids is None:
                col_ids = self._ids_to_col_ids(input_ids)
            token_embs = (
                token_embs + self.row_emb(row_ids) + self.col_emb(col_ids)
            )

            # Combined conditioning: pokemon identity + metadata
            B, device = token_embs.shape[0], token_embs.device

            def _rand_or_use(val, emb):
                if val is None:
                    val = torch.randint(
                        0,
                        emb.num_embeddings,
                        (B,),
                        device=device,
                    )
                return emb(val)

            # pokemon_cond allows passing a pre-computed conditioning vector directly
            # (used for novel Pokémon generation via embedding interpolation)
            base_cond = (
                pokemon_cond
                if pokemon_cond is not None
                else self.conditioning(pokemon_idx)
            )
            cond = (
                base_cond
                + _rand_or_use(type1, self.type1_emb)
                + _rand_or_use(type2, self.type2_emb)
                + _rand_or_use(is_shiny, self.is_shiny_emb)
                + _rand_or_use(generation, self.generation_emb)
                + _rand_or_use(evolution_stage, self.evo_stage_emb)
                + _rand_or_use(has_evolution, self.has_evolution_emb)
                + _rand_or_use(color_shift, self.color_shift_emb)
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
            shift_labels = (
                labels[..., 1:].contiguous().to(outputs.logits.device)
            )
            weights = self.token_weights.to(outputs.logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                weight=weights,
                ignore_index=-100,
                reduction="mean",
            )
            outputs.loss = torch.nan_to_num(loss, nan=0.0)

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        **kwargs,
    ):
        # Compute positional ids from the full sequence before the parent trims for KV cache
        row_ids_full = self._ids_to_row_ids(input_ids)
        col_ids_full = self._ids_to_col_ids(input_ids)

        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

        # Trim to last token when KV cache is active (same logic as parent trims input_ids)
        if past_key_values is not None:
            inputs["row_ids"] = row_ids_full[:, -1:]
            inputs["col_ids"] = col_ids_full[:, -1:]
        else:
            inputs["row_ids"] = row_ids_full
            inputs["col_ids"] = col_ids_full

        for field in (
            "pokemon_idx",
            "pokemon_cond",
            "type1",
            "type2",
            "is_shiny",
            "generation",
            "evolution_stage",
            "has_evolution",
            "color_shift",
        ):
            if field in kwargs:
                inputs[field] = kwargs[field]

        return inputs

    def sample_conditioning(self, idx: int | None = None) -> torch.Tensor:
        if idx is None:
            idx = torch.randint(
                0,
                self.conditioning.num_embeddings,
                (1,),
            ).item()
        with torch.no_grad():
            return self.conditioning(
                torch.tensor([idx], device=self.conditioning.weight.device),
            )

    def sample_random_conditioning(self, device: str = "cpu") -> dict:
        return {
            "pokemon_idx": torch.randint(
                0,
                self.conditioning.num_embeddings,
                (1,),
                device=device,
            ),
            "type1": torch.randint(
                0,
                self.type1_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "type2": torch.randint(
                0,
                self.type2_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "is_shiny": torch.randint(
                0,
                self.is_shiny_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "generation": torch.randint(
                0,
                self.generation_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "evolution_stage": torch.randint(
                0,
                self.evo_stage_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "has_evolution": torch.randint(
                0,
                self.has_evolution_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "color_shift": torch.randint(
                0,
                NUM_COLOR_SHIFTS,
                (1,),
                dtype=torch.long,
                device=device,
            ),
        }

    def sample_novel_conditioning(
        self,
        n_mix: int = 3,
        device: str = "cpu",
    ) -> dict:
        """Blend n_mix random Pokémon embeddings to produce a novel conditioning vector."""
        with torch.no_grad():
            idxs = torch.randint(
                0,
                self.conditioning.num_embeddings,
                (n_mix,),
                device=device,
            )
            weights = torch.softmax(torch.randn(n_mix, device=device), dim=0)
            pokemon_cond = (weights.unsqueeze(1) * self.conditioning(idxs)).sum(
                0,
                keepdim=True,
            )
        return {
            "pokemon_cond": pokemon_cond,
            "type1": torch.randint(
                0,
                self.type1_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "type2": torch.randint(
                0,
                self.type2_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "is_shiny": torch.randint(
                0,
                self.is_shiny_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "generation": torch.randint(
                0,
                self.generation_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "evolution_stage": torch.randint(
                0,
                self.evo_stage_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "has_evolution": torch.randint(
                0,
                self.has_evolution_emb.num_embeddings,
                (1,),
                device=device,
            ),
            "color_shift": torch.randint(
                0,
                NUM_COLOR_SHIFTS,
                (1,),
                dtype=torch.long,
                device=device,
            ),
        }

    def generate_sprite(
        self,
        tokenizer,
        temperature: float = 1.2,
        top_p: float = 0.95,
        verbose: bool = False,
        type1: "str | int | None" = None,
        type2: "str | int | None" = None,
    ) -> np.ndarray:
        """Generate a 64×64 RGB sprite as a numpy array.

        type1 / type2 accept a type name ("fire", "water", …) or its integer
        index. Pass None to pick a random value.
        """
        device = next(self.parameters()).device

        cond = self.sample_novel_conditioning(device=str(device))
        cond["color_shift"] = torch.tensor([0], dtype=torch.long, device=device)
        cond["is_shiny"] = torch.tensor([0], dtype=torch.long, device=device)
        cond["generation"] = torch.tensor(
            [2 + torch.randint(0, 2, (1,)).item()],
            dtype=torch.long,
            device=device,
        )

        t1 = _resolve_type(type1, _TYPE1_UNK)
        t2 = _resolve_type(type2, _TYPE2_NONE)
        if t1 is not None:
            cond["type1"] = torch.tensor([t1], dtype=torch.long, device=device)
        if t2 is not None:
            cond["type2"] = torch.tensor([t2], dtype=torch.long, device=device)

        row_marker_ids = [
            tokenizer.convert_tokens_to_ids(f"[ROW_{i:02d}]") for i in range(64)
        ]
        inputs = tokenizer(
            "[ROW_00]",
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs.update(cond)

        streamer = None
        if verbose:
            from transformers import TextStreamer

            class _CompactStreamer(TextStreamer):
                def on_finalized_text(
                    self, text: str, stream_end: bool = False
                ):
                    prefix = "\n" if "[ROW" in text else ""
                    print(
                        prefix + text,
                        end="" if not stream_end else "\n",
                        flush=True,
                    )

            streamer = _CompactStreamer(tokenizer, skip_special_tokens=False)

        with torch.no_grad():
            output_ids = self.generate(
                **inputs,
                max_length=4096,
                do_sample=True,
                top_k=0,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=[
                    _RowLengthLogitsProcessor(tokenizer, row_marker_ids),
                ],
                streamer=streamer,
            )

        return _tokens_to_image(
            tokenizer.decode(output_ids[0], skip_special_tokens=False),
        )


class _RowLengthLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        row_marker_ids: list[int],
        row_width: int = 63,
    ):
        vocab = tokenizer.get_vocab()
        self.row_marker_set = set(row_marker_ids)
        self.row_marker_ids = row_marker_ids
        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id
        self.row_width = row_width
        special = {
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.pad_token_id,
            *row_marker_ids,
        }
        self.pixel_len = torch.zeros(len(vocab), dtype=torch.long)
        for token, idx in vocab.items():
            if idx not in special:
                self.pixel_len[idx] = len(token)
        self.current_row = self.chars_in_row = 0

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        device = scores.device
        pixel_len = self.pixel_len.to(device)
        last_id = int(input_ids[0, -1].item())
        if last_id in self.row_marker_set:
            self.current_row = self.row_marker_ids.index(last_id)
            self.chars_in_row = 0
        remaining = self.row_width - self.chars_in_row
        if remaining > 0:
            mask = (pixel_len > remaining) | (pixel_len == 0)
            if self.current_row < 63:
                mask[self.eos_id] = True
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        else:
            allowed = torch.full((scores.shape[-1],), True, device=device)
            if self.current_row < 63:
                allowed[self.row_marker_ids[self.current_row + 1]] = False
            else:
                allowed[self.eos_id] = False
            scores = scores.masked_fill(allowed.unsqueeze(0), float("-inf"))
        if last_id not in self.row_marker_set and last_id not in {
            self.eos_id,
            self.bos_id,
        }:
            self.chars_in_row += int(pixel_len[last_id].item())
        return scores


def _tokens_to_image(text: str) -> np.ndarray:
    rows, current = [], []
    for token in text.split():
        if token.startswith("[ROW_") and token.endswith("]"):
            if current:
                rows.append(current)
            current = []
        elif not (token.startswith("[") and token.endswith("]")):
            current.append(token)
    if current:
        rows.append(current)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    for y, row in enumerate(rows[:64]):
        for x, char in enumerate(row[:64]):
            if char == "~":
                image[y, x] = [255, 255, 255]
            else:
                idx = ord(char) - 59
                r, g, b = idx // 16, (idx // 4) % 4, idx % 4
                image[y, x] = [r * 64 + 32, g * 64 + 32, b * 64 + 32]
    return image


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    REPO_ID = "iamthinbaker/GPokeT2"
    REVISION = "v0.1-wip-3400"
    N_SAMPLES = 5
    OUTPUT_DIR = Path("generated")

    print(f"Descargando {REPO_ID} ({REVISION})...")
    ckpt = snapshot_download(REPO_ID, revision=REVISION)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
    model.eval()

    OUTPUT_DIR.mkdir(exist_ok=True)
    for i in range(N_SAMPLES):
        print(f"\n[{i + 1}/{N_SAMPLES}] generando...")
        image = model.generate_sprite(tokenizer, verbose=True)
        path = OUTPUT_DIR / f"pokemon_{i + 1:02d}.png"
        Image.fromarray(image).save(path)
        print(f"guardado en {path}")
