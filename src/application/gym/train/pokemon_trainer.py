import tempfile
from functools import partial

import torch
from transformers import Trainer  # type: ignore
from transformers import GPT2Config
from transformers import TrainingArguments  # type: ignore

from src.domain.gld.prof_oak_pc import BoxEntity
from src.application.gym.model import ConditionedGPT2
from src.application.gym.model import ConditionedDataCollator
from src.application.gym.model import ForCausalLMLossWeighed
from src.application.gym.train.callbacks import CheckpointStorageCallback
from src.application.gym.train.callbacks import InferenceCallback


class PokemonTrainer:
    def __init__(
        self,
        checkpoint_storage_callback: CheckpointStorageCallback,
        context_length=4096,
        row_length=64,
    ):
        self.row_length = row_length
        self.context_length = context_length
        self.checkpoint_storage_callback = checkpoint_storage_callback

    def train(
        self,
        box_entity: BoxEntity,
    ):
        dataset = box_entity.dataset
        tokenizer = box_entity.tokenizer
        num_pokemon = getattr(
            tokenizer,
            "num_pokemon",
            len(dataset["train"].unique("pokemon_idx")),
        )

        self.inference_callback = InferenceCallback(
            context_length=self.context_length,
            row_length=self.row_length,
            interval_steps=50,
            tokenizer=tokenizer,
        )

        data_collator = ConditionedDataCollator(
            tokenizer=tokenizer,
            mlm=False,
        )

        model = ConditionedGPT2(
            config=GPT2Config(
                vocab_size=len(tokenizer.get_vocab()),
                n_ctx=self.context_length,
                n_positions=self.context_length,
                n_embd=256,
                n_layer=6,
                n_head=4,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ),
            num_pokemon=num_pokemon,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer_args = TrainingArguments(
                output_dir=tmpdirname,
                per_device_train_batch_size=32,
                num_train_epochs=1000,
                logging_steps=10,
                gradient_accumulation_steps=16,
                save_strategy="steps",
                save_steps=100,
                learning_rate=5e-4,
                weight_decay=0.1,
                warmup_ratio=0.05,
                bf16=torch.cuda.is_available(),
                dataloader_pin_memory=torch.cuda.is_available(),
                dataloader_num_workers=4,
                optim="adamw_torch_fused",
                torch_compile=torch.cuda.is_available(),
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                args=trainer_args,
                data_collator=data_collator,
                train_dataset=dataset["train"],
                compute_loss_func=partial(
                    ForCausalLMLossWeighed,
                    vocab_size=len(tokenizer.get_vocab()),
                    weight_token_id=tokenizer.convert_tokens_to_ids("~"),
                    token_weight=0.3,
                ),
                callbacks=[
                    self.inference_callback,
                    self.checkpoint_storage_callback,
                ],
            )

            trainer.train(
                resume_from_checkpoint=self.checkpoint_storage_callback.resume_from_checkpoint,
            )

        return self
