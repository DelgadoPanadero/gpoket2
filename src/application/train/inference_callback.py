import math
import torch

from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast


class InferenceCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        row_length: int = 64,
        context_length : int = 1024,
        interval_steps : int = 5000,
    ):
        self.interval_steps = interval_steps
        self.tokenizer = tokenizer
        self.row_length = row_length
        self.context_length = context_length


    def _increase_inference_context(
        self,
        model:GPT2LMHeadModel,
        new_context_length:int
    )->GPT2LMHeadModel:
        
        # Update context parameters
        model.config.n_ctx = new_context_length
        model.config.n_positions = new_context_length
        
        # Get current attention weights
        old_embed = model.transformer.wpe.weight.data
        old_max_positions, emb_dim = old_embed.shape

        # Tiled attention until new context length
        repeats = math.ceil(new_context_length / old_max_positions)
        tiled_embed = old_embed.repeat((repeats, 1))[: new_context_length]
        model.transformer.wpe.weight = torch.nn.Parameter(tiled_embed)

        return model
    
    def _generation(
        self,
        model: GPT2LMHeadModel,
        input_text: str,
    )->str:

        device = next(model.parameters()).device
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=self.context_length,
                min_length=self.context_length,
                do_sample=False,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        decoded = self.tokenizer.decode(
            output[0], skip_special_tokens=False
        )
        return decoded


    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (
            state.global_step % self.interval_steps == 0
            and state.global_step > 0
        ):
            model: GPT2LMHeadModel = kwargs["model"]

            model = self._increase_inference_context(
                model=model,
                new_context_length=self.context_length,
            )

            input_text = "00"

            decoded = self._generation(
                model = model,
                input_text=input_text,
            )

            print(f"\n\n=== Inference @ step {state.global_step} ===")
            print(decoded.replace(f"00", "\n00"))
            print("====================================\n\n")

        return control
