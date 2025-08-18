import torch

from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback


class InferenceCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        interval_steps=5000,
        prompt="[BOS]",
        max_length=512,
    ):
        self.interval_steps = interval_steps
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length

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
            model = kwargs["model"]
            device = next(model.parameters()).device

            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(device)
            model.eval()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=self.max_length,
                    min_length=self.max_length,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\n\n=== Inference @ step {state.global_step} ===")
            print(decoded)
            print("====================================\n\n")

        return control
