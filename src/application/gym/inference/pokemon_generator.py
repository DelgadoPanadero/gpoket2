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
        num_pokemon = state_dict["conditioning.weight"].shape[0]

        self.model = ConditionedGPT2(
            config=config,
            num_pokemon=num_pokemon,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.tie_weights()
        self.model.to(self.device)
        self.model.eval()
        self.num_pokemon = num_pokemon
        self.pokemon_repository = pokemon_repository

    def _text_to_image(self, text: str) -> np.ndarray:
        pixels = [t for t in text.split() if len(t) == 1]
        rows = [
            pixels[i : i + self._IMAGE_WIDTH]
            for i in range(
                0, self._IMAGE_WIDTH * self._IMAGE_WIDTH, self._IMAGE_WIDTH
            )
        ]
        for row in rows:
            row += ["~"] * (self._IMAGE_WIDTH - len(row))

        return PokemonEncoder._decode(rows)

    def generate(
        self,
        pokemon_idx: int | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> tuple[str, int]:
        if pokemon_idx is None:
            pokemon_idx = torch.randint(0, self.num_pokemon, (1,)).item()

        inputs = self.tokenizer(self.tokenizer.bos_token, return_tensors="pt")
        inputs["pokemon_idx"] = torch.tensor([pokemon_idx], dtype=torch.long)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

        entity = PokemonEntity(
            name=f"{pokemon_idx:04d}.png",
            generation="generated",
            game_name="generated",
            image=image_bytes.tobytes(),
        )
        saved_path = self.pokemon_repository.save_one(entity)
        return saved_path, pokemon_idx
