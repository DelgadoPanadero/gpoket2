from pathlib import Path

import cv2
import torch
import numpy as np
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast, GPT2Config

from src.domain.brz.pokemon import PokemonEntity
from src.domain.brz.pokemon import PokemonRepository
from src.application.slv.pokedex.encoder import PokemonEncoder
from src.application.gym.model.conditioned_gpt2 import ConditionedGPT2
from src.application.gym.model.sprite_validator import SpriteValidator
from src.application.gym.inference.row_length_logits_processor import (
    RowLengthLogitsProcessor,
)
from src.domain.brz.pokemon.pokemon_metadata import (
    EvolutionStage,
    PokemonType,
    Shininess,
)
from src.application.gld.prof_oak_pc.metadata_adapter import TYPE2_NONE


class PokemonGenerator:
    _IMAGE_WIDTH = 64
    _CONTEXT_LENGTH = 4096
    _ROW_PREFIX = "[ROW_"

    def __init__(
        self,
        checkpoint_path: str | Path,
        pokemon_repository: PokemonRepository,
        validator_path: str | Path | None = None,
        device: str | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if device == "gpu" else device

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(checkpoint_path),
        )

        config = GPT2Config.from_pretrained(str(checkpoint_path))
        state_dict = load_file(checkpoint_path / "model.safetensors")

        # num_pokemon is inferred from the conditioning embedding in the saved state
        num_pokemon = state_dict["conditioning.weight"].shape[0]
        self.model = ConditionedGPT2(config=config, num_pokemon=num_pokemon)
        self.model.load_state_dict(state_dict, strict=False)

        # load_state_dict silently skips None-initialized buffers (PyTorch filters
        # them out before copy_), so we manually restore them from the state dict.
        if (
            getattr(self.model, "token_weights", None) is None
            and "token_weights" in state_dict
        ):
            setattr(self.model, "token_weights", state_dict["token_weights"])

        self.model.tie_weights()
        self.model.to(self.device)
        self.model.eval()

        self.pokemon_repository = pokemon_repository

        self.validator: SpriteValidator | None = None
        if validator_path is not None and Path(validator_path).exists():
            self.validator = SpriteValidator.load(
                validator_path,
                device=self.device,
            )

        row_marker_ids = [
            self.tokenizer.convert_tokens_to_ids(f"[ROW_{i:02d}]")
            for i in range(64)
        ]
        self.row_processor = RowLengthLogitsProcessor(
            self.tokenizer,
            row_marker_ids,
            row_width=63,
        )

    def _text_to_image(self, text: str) -> np.ndarray:
        """
        Reconstruct a 64×64 image from character-level decoded text.
        Rows are delimited by [ROW_XX] tokens. Each non-special token is one pixel.
        """
        rows: list[list[str]] = []
        current: list[str] = []

        for token in text.split():
            if token.startswith(self._ROW_PREFIX) and token.endswith("]"):
                if current:
                    rows.append(current)
                current = []
            elif token.startswith("[") and token.endswith("]"):
                pass  # skip [BOS], [EOS], [PAD], [UNK]
            else:
                current.append(token)  # each token is exactly one pixel

        if current:
            rows.append(current)

        image_rows: list[list[str]] = []
        for row in rows[: self._IMAGE_WIDTH]:
            full_row = row[: self._IMAGE_WIDTH]
            full_row += ["~"] * (self._IMAGE_WIDTH - len(full_row))
            image_rows.append(full_row)

        while len(image_rows) < self._IMAGE_WIDTH:
            image_rows.append(["~"] * self._IMAGE_WIDTH)

        return PokemonEncoder._decode(image_rows)

    def _generate_one(
        self,
        cond: dict,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> tuple[np.ndarray, dict]:
        inputs = self.tokenizer(
            "[ROW_00]",
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs.update(cond)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self._CONTEXT_LENGTH,
                do_sample=True,
                top_k=0,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                logits_processor=[self.row_processor],
            )

        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False,
        )
        image = self._text_to_image(full_text)

        if "pokemon_idx" in cond:
            cond_meta = {"pokemon_idx": int(cond["pokemon_idx"].item())}
        else:
            cond_meta = {"pokemon_idx": None}
        return image, cond_meta

    def generate(
        self,
        pokemon_idx: int | None = None,
        type1: int | None = None,
        type2: int | None = None,
        is_shiny: int | None = None,
        generation: int | None = None,
        evolution_stage: int | None = None,
        has_evolution: int | None = None,
        novel: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
        n_candidates: int = 3,
        min_score: float = 0.55,
    ) -> tuple[str, dict]:
        if pokemon_idx is not None:
            # Specific existing Pokémon requested
            cond = self.model.sample_random_conditioning(device=self.device)
            cond["pokemon_idx"] = torch.tensor(
                [pokemon_idx],
                dtype=torch.long,
                device=self.device,
            )
        elif novel:
            cond = self.model.sample_novel_conditioning(device=self.device)
        else:
            cond = self.model.sample_random_conditioning(device=self.device)

        # Always use original palette, non-shiny form and gen3/gen4 (0-indexed: 2 or 3)
        cond["color_shift"] = torch.tensor(
            [0],
            dtype=torch.long,
            device=self.device,
        )
        cond["is_shiny"] = torch.tensor(
            [0],
            dtype=torch.long,
            device=self.device,
        )
        cond["generation"] = torch.tensor(
            [2 + torch.randint(0, 2, (1,)).item()],
            dtype=torch.long,
            device=self.device,
        )

        # Override any metadata fields explicitly provided
        _override = {
            "type1": type1,
            "type2": type2,
            "is_shiny": is_shiny,
            "generation": generation,
            "evolution_stage": evolution_stage,
            "has_evolution": has_evolution,
        }
        for field, val in _override.items():
            if val is not None:
                cond[field] = torch.tensor(
                    [val],
                    dtype=torch.long,
                    device=self.device,
                )

        best_image: np.ndarray | None = None
        best_meta: dict = {}
        best_score: float = -1.0

        for _ in range(n_candidates):
            image, meta = self._generate_one(
                cond,
                temperature=temperature,
                top_p=top_p,
            )
            score = self.validator.score(image) if self.validator else 1.0

            if score > best_score:
                best_score = score
                best_image = image
                best_meta = meta

            if best_score >= min_score:
                break

        assert best_image is not None
        _, image_bytes = cv2.imencode(".png", best_image)

        # Extract actual conditioning values used (random or explicitly provided)
        cond_values: dict = {}
        for field in (
            "pokemon_idx",
            "type1",
            "type2",
            "is_shiny",
            "generation",
            "evolution_stage",
            "has_evolution",
        ):
            if field in cond:
                cond_values[field] = int(cond[field].item())

        t1_idx = cond_values.get("type1", len(PokemonType))
        t1 = (
            PokemonType(t1_idx).name.lower()
            if t1_idx < len(PokemonType)
            else "unk"
        )
        t2_idx = cond_values.get("type2", TYPE2_NONE)
        t2 = (
            PokemonType(t2_idx).name.lower()
            if t2_idx < len(PokemonType)
            else None
        )
        type_str = f"{t1}-{t2}" if t2 else t1

        name_parts = ["pokemon"]
        if "pokemon_idx" in cond_values:
            name_parts.append(str(cond_values["pokemon_idx"]))
        name_parts.append(type_str)
        name_parts.append(f"sh{cond_values.get('is_shiny', 0)}")
        name_parts.append(f"g{cond_values.get('generation', 0) + 1}")
        name_parts.append(f"ev{cond_values.get('evolution_stage', 0)}")
        name_parts.append(f"he{cond_values.get('has_evolution', 0)}")
        filename = "_".join(name_parts) + ".png"

        evo_raw = cond_values.get("evolution_stage")
        entity = PokemonEntity(
            name=filename,
            generation="generated",
            game_name="generated",
            image=image_bytes.tobytes(),
            type_1=PokemonType(t1_idx) if t1_idx < len(PokemonType) else None,
            type_2=PokemonType(t2_idx) if t2_idx < len(PokemonType) else None,
            shininess=Shininess(cond_values.get("is_shiny", 0)),
            evolution_stage=EvolutionStage(
                min(evo_raw, len(EvolutionStage) - 1),
            )
            if evo_raw is not None
            else None,
            has_evolution=bool(cond_values.get("has_evolution", 0)),
        )
        saved_path = self.pokemon_repository.save_one(entity)
        return saved_path, {**cond_values, "validator_score": best_score}
