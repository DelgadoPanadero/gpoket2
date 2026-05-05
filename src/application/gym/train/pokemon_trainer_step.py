import tempfile
from functools import partial

import torch
from transformers import Trainer  # type: ignore
from transformers import GPT2Config
from transformers import TrainingArguments  # type: ignore

from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.gld.prof_oak_pc import ProfOakPcRepository
from src.application.gym.model import ConditionedGPT2
from src.application.gym.model import ConditionedDataCollator
from src.application.gym.model import ForCausalLMLossWeighed
from src.application.gym.train.callbacks import CheckpointStorageCallback
from src.application.gym.train.callbacks import InferenceCallback


class PokemonTrainerStep:
    def __init__(
        self,
        profoakpc_repository: ProfOakPcRepository,
        checkpoint_storage_adapter,
        context_length=1024,
        row_length=64,
    ):
        self.row_length = row_length
        self.context_length = context_length
        self.profoakpc_repository = profoakpc_repository
        self.checkpoint_storage_adapter = checkpoint_storage_adapter

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
            context_length=self.row_length * self.row_length,
            row_length=self.row_length,
            interval_steps=100,
            tokenizer=tokenizer,
        )

        self.checkpoint_storage_callback = CheckpointStorageCallback(
            checkpoint_storage_adapter=self.checkpoint_storage_adapter
        )
        data_collator = ConditionedDataCollator(
            tokenizer=tokenizer,
            mlm=False,
        )

        _vocab = tokenizer.get_vocab()
        vocab_size = (
            max(_vocab.values()) + 1
        )  # +1 accounts for holes in saved vocab

        model = ConditionedGPT2(
            config=GPT2Config(
                vocab_size=vocab_size,
                n_ctx=self.context_length,
                n_positions=self.context_length,
                n_embd=512,
                n_layer=8,
                n_head=8,
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
                num_train_epochs=50,
                logging_steps=50,
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
            )

            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                args=trainer_args,
                data_collator=data_collator,
                train_dataset=dataset["train"],
                compute_loss_func=partial(
                    ForCausalLMLossWeighed,
                    vocab_size=vocab_size,
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

    def run(
        self,
    ):
        box_entity = self.profoakpc_repository.load()
        return self.train(box_entity)
