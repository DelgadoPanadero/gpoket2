import re
import torch
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast, GPT2Config

from src.application.shared import PokemonEncoder
from src.application.train.conditioned_gpt2 import ConditionedGPT2


class PokemonGenerator:
    _ROW_RE = re.compile(r"^\d{2}$")
    _IMAGE_WIDTH = 64
    _CONTEXT_LENGTH = 4096

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if device == "gpu" else device

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

    def _text_to_image(self, text: str) -> Image.Image:
        rows: list[list[str]] = []
        current: list[str] = []
        for token in text.split():
            if self._ROW_RE.match(token):
                if current:
                    rows.append(current)
                # col 0 was dropped during encoding; restore as blank
                current = ["~"]
            elif len(token) == 1:
                current.append(token)
        if current:
            rows.append(current)

        if not rows:
            rows = [["~"] * self._IMAGE_WIDTH for _ in range(self._IMAGE_WIDTH)]

        bgr = PokemonEncoder._decode(rows)
        return Image.fromarray(bgr[:, :, ::-1])

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
