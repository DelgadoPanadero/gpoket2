import copy
import math
import torch
import threading

from transformers import TrainerState  # type: ignore
from transformers import TrainerControl  # type: ignore
from transformers import TrainerCallback  # type: ignore
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments  # type: ignore
from transformers import PreTrainedTokenizerFast  # type: ignore


class InferenceCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        device: str = "gpu",
        syncronous: bool = True,
        row_length: int = 64,
        context_length: int = 4096,
        interval_steps: int = 100,
    ):
        self.interval_steps = interval_steps
        self.device = device
        self.syncronous = syncronous
        self.tokenizer = tokenizer
        self.row_length = row_length
        self.context_length = context_length

    def _increase_inference_context(
        self,
        model: GPT2LMHeadModel,
        new_context_length: int,
    ) -> GPT2LMHeadModel:
        # Update context parameters
        model.config.n_ctx = new_context_length
        model.config.n_positions = new_context_length

        # Get current attention weights
        old_embed = model.transformer.wpe.weight.data
        old_max_positions, emb_dim = old_embed.shape

        # Tiled attention until new context length
        repeats = math.ceil(new_context_length / old_max_positions)
        tiled_embed = old_embed.repeat((repeats, 1))[:new_context_length]
        model.transformer.wpe.weight = torch.nn.Parameter(tiled_embed)

        return model

    def _generation(
        self,
        model: GPT2LMHeadModel,
        input_text: dict,
        step: int = 0,
    ) -> None:
        with torch.no_grad():
            output = model.generate(
                **input_text,
                max_length=self.context_length,
                min_length=self.context_length,
                do_sample=True,
                top_k=5,  # our dataset is very small
                top_p=0.95,
                temperature=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=False)

        print(f"\n\n=== Inference @ step {step} ===")
        print(decoded)
        print("====================================\n\n")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (
            state.global_step % self.interval_steps == 0
            and state.global_step > 0
        ):
            model: GPT2LMHeadModel = kwargs["model"]

            inference_model = copy.deepcopy(model)
            inference_model = self._increase_inference_context(
                model=inference_model,
                new_context_length=self.context_length,
            )

            device = "cuda" if self.device == "gpu" else "cpu"

            input_text = self.tokenizer("00", return_tensors="pt").to(device)

            # Pass pokemon_idx=0 during training monitoring to track learning progress
            if hasattr(inference_model, "conditioning"):
                input_text["pokemon_idx"] = torch.zeros(
                    1,
                    dtype=torch.long,
                    device=device,
                )

            if self.device == "cpu":
                inference_model = inference_model.cpu().half()

            inference_model.eval()

            if self.syncronous:
                self._generation(
                    model=inference_model,
                    input_text=input_text,
                    step=state.global_step,
                )

            else:
                thread = threading.Thread(
                    target=self._generation,
                    args=(inference_model, input_text, state.global_step),
                    daemon=True,
                )
                thread.start()
