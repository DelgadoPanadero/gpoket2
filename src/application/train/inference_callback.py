import torch

from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback
from transformers import GPT2LMHeadModel


class InferenceCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        interval_steps=5000,
        row_length=64,
        context_length=1024,
    ):
        self.interval_steps = interval_steps
        self.tokenizer = tokenizer
        self.row_length = row_length
        self.context_length = context_length

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

            model.config.n_positions = self.context_length
            model.config.n_ctx = self.context_length
            old_embed = model.transformer.wpe.weight.data
            old_max_positions, emb_dim = old_embed.shape
            repeats = (
                self.context_length + old_max_positions - 1
            ) // old_max_positions
            tiled_embed = old_embed.repeat((repeats, 1))[: self.context_length]
            model.transformer.wpe.weight = torch.nn.Parameter(tiled_embed)
            # model.eval()

            def run(inputs):
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
                return output

            text = "[BOS]"
            device = next(model.parameters()).device
            inputs = self.tokenizer(text, return_tensors="pt").to(device)
            output = run(inputs=inputs)

            decoded = self.tokenizer.decode(
                output[0], skip_special_tokens=False
            )
            print(f"\n\n=== Inference @ step {state.global_step} ===")

            try:
                print(
                    "\n".join(
                        [
                            " ".join(
                                decoded.split(" ")[i : i + self.row_length]
                            )
                            for i in range(
                                0, self.context_length, self.row_length
                            )
                        ]
                    )
                )

            except:
                print(decoded)
            print("====================================\n\n")

        return control
