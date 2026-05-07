import cv2
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast, GPT2Config

from src.domain.brz.pokemon import PokemonEntity
from src.domain.brz.pokemon import PokemonRepository
from src.application.slv.pokedex.encoder import PokemonEncoder
from src.application.gym.model.conditioned_gpt2 import ConditionedGPT2
from src.application.gld.prof_oak_pc.metadata_adapter import get_name_chars


class PokemonGenerator:
    _IMAGE_WIDTH = 64
    _CONTEXT_LENGTH = 4096

    def __init__(
        self,
        checkpoint_path: str | Path,
        pokemon_repository: PokemonRepository,
        device: str | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if device == "gpu" else device

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(checkpoint_path)
        )

        config = GPT2Config.from_pretrained(str(checkpoint_path))
        state_dict = load_file(checkpoint_path / "model.safetensors")

        self.model = ConditionedGPT2(config=config)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.tie_weights()
        self.model.to(self.device)
        self.model.eval()
        self.pokemon_repository = pokemon_repository

    def _text_to_image(self, text: str) -> np.ndarray:
        rows: list[list[str]] = []
        current: list[str] = []
        for token in text.split():
            if len(token) == 2 and token.isdigit():
                if current:
                    rows.append(current)
                current = []
            elif len(token) == 1:
                current.append(token)
        if current:
            rows.append(current)

        image_rows = []
        for row in rows[: self._IMAGE_WIDTH]:
            # first pixel of each row was dropped during tokenization; replace with background
            full_row = ["~"] + row[: self._IMAGE_WIDTH - 1]
            full_row += ["~"] * (self._IMAGE_WIDTH - len(full_row))
            image_rows.append(full_row)
        while len(image_rows) < self._IMAGE_WIDTH:
            image_rows.append(["~"] * self._IMAGE_WIDTH)

        return PokemonEncoder._decode(image_rows)

    def generate(
        self,
        name: str | None = None,
        type1: int | None = None,
        type2: int | None = None,
        is_shiny: int | None = None,
        generation: int | None = None,
        evolution_stage: int | None = None,
        has_evolution: int | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> tuple[str, dict]:
        cond = self.model.sample_random_conditioning(device=self.device)

        if name is not None:
            chars = get_name_chars(name + ".txt")
            cond["name_chars"] = torch.tensor(
                [chars], dtype=torch.long, device=self.device
            )

        overrides = {
            "type1": type1,
            "type2": type2,
            "is_shiny": is_shiny,
            "generation": generation,
            "evolution_stage": evolution_stage,
            "has_evolution": has_evolution,
        }
        for key, val in overrides.items():
            if val is not None:
                cond[key] = torch.tensor(
                    [val], dtype=torch.long, device=self.device
                )

        inputs = self.tokenizer(self.tokenizer.bos_token, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs.update(cond)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self._CONTEXT_LENGTH,
                min_length=self._CONTEXT_LENGTH,
                do_sample=True,
                top_k=0,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=False
        )
        image = self._text_to_image(full_text)
        _, image_bytes = cv2.imencode(".png", image)

        cond_meta = {
            "type1": int(cond["type1"].item()),
            "type2": int(cond["type2"].item()),
            "is_shiny": int(cond["is_shiny"].item()),
            "generation": int(cond["generation"].item()),
        }
        entity = PokemonEntity(
            name=f"t{cond_meta['type1']}_t{cond_meta['type2']}_s{cond_meta['is_shiny']}_g{cond_meta['generation']}.png",
            generation="generated",
            game_name="generated",
            image=image_bytes.tobytes(),
        )
        saved_path = self.pokemon_repository.save_one(entity)
        return saved_path, cond_meta
