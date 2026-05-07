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
                top_k=0,
                top_p=0.95,
                temperature=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

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

                input_text = self.tokenizer("00", return_tensors="pt").to(device)

                if hasattr(base_model, "conditioning"):
                    num_pokemon = base_model.conditioning.num_embeddings
                    pokemon_idx = state.global_step % num_pokemon
                    input_text["pokemon_idx"] = torch.tensor(
                        [pokemon_idx],
                        dtype=torch.long,
                        device=device,
                    )

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
                print(f"\n[InferenceCallback] Error @ step {state.global_step}: {e}\n", flush=True)
