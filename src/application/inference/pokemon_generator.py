import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast, GPT2Config

from src.application.train.conditioned_gpt2 import ConditionedGPT2


class PokemonGenerator:
    _ROW_RE = re.compile(r"^\d{2}$")
    _BLANK = "~"
    _WHITE = (255, 255, 255)
    _IMAGE_WIDTH = 64
    _CONTEXT_LENGTH = 4096

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(checkpoint_path))

        config = GPT2Config.from_pretrained(str(checkpoint_path))
        state_dict = load_file(checkpoint_path / "model.safetensors")
        num_pokemon = state_dict["conditioning.weight"].shape[0]

        self.model = ConditionedGPT2(config=config, num_pokemon=num_pokemon)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.tie_weights()
        self.model.to(self.device)
        self.model.eval()
        self.num_pokemon = num_pokemon

    def _char_to_rgb(self, char: str) -> tuple[int, int, int]:
        if char == self._BLANK:
            return self._WHITE
        idx = ord(char) - 59
        r, g, b = idx // 16, (idx % 16) // 4, idx % 4
        return (r * 85, g * 85, b * 85)

    def _text_to_image(self, text: str) -> Image.Image:
        rows: list[list[str]] = []
        current: list[str] = []
        for token in text.split():
            if self._ROW_RE.match(token):
                if current:
                    rows.append(current)
                current = []
            elif len(token) == 1:
                current.append(token)
        if current:
            rows.append(current)

        height = len(rows) or self._IMAGE_WIDTH
        img = np.full((height, self._IMAGE_WIDTH, 3), 255, dtype=np.uint8)
        for y, row in enumerate(rows):
            for x, char in enumerate(row[: self._IMAGE_WIDTH - 1]):
                img[y, x + 1] = self._char_to_rgb(char)

        return Image.fromarray(img)

    def generate(
        self,
        pokemon_idx: int | None = None,
        temperature: float = 1.2,
        top_p: float = 0.95,
    ) -> tuple[Image.Image, int]:
        if pokemon_idx is None:
            pokemon_idx = torch.randint(0, self.num_pokemon, (1,)).item()

        inputs = self.tokenizer("00", return_tensors="pt")
        inputs["pokemon_idx"] = torch.tensor([pokemon_idx], dtype=torch.long)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._text_to_image(text), pokemon_idx
