import torch

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
        context_length: int = 4096,
        interval_steps: int = 100,
        max_new_tokens: int | None = None,
    ):
        self.interval_steps = interval_steps
        self.device = device
        self.syncronous = syncronous
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else context_length - 1
        )

    def _generation(
        self,
        model: GPT2LMHeadModel,
        input_text: dict,
        step: int = 0,
    ) -> None:
        input_length = input_text["input_ids"].shape[1]
        max_new_tokens = min(
            self.max_new_tokens,
            self.context_length - input_length - 1,
        )
        if max_new_tokens <= 0:
            print(
                f"\n[InferenceCallback] Skipping inference @ step {step}: input length {input_length} >= context length {self.context_length}\n",
                flush=True,
            )
            return

        with torch.inference_mode():
            torch.cuda.synchronize()
            output = model.generate(
                **input_text,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=False)

        print(f"\n\n=== Inference @ step {step} ===", flush=True)
        print(decoded, flush=True)
        print("====================================\n\n", flush=True)

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
            try:
                model: GPT2LMHeadModel = kwargs["model"]
                base_model = getattr(model, "_orig_mod", model)

                device = "cuda" if self.device == "gpu" else "cpu"

                input_text = self.tokenizer(
                    "[ROW_00]",
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(device)

                if hasattr(base_model, "sample_random_conditioning"):
                    cond = base_model.sample_random_conditioning(device=device)
                    input_text.update(cond)

                base_model.eval()
                try:
                    self._generation(
                        model=base_model,
                        input_text=input_text,
                        step=state.global_step,
                    )
                finally:
                    base_model.train()

            except Exception as e:
                print(
                    f"\n[InferenceCallback] Error @ step {state.global_step}: {e}\n",
                    flush=True,
                )
