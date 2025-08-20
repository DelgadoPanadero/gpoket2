import tempfile

import torch
from transformers import Trainer  # type: ignore
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore

from src.domain.gld.prof_oak_pc import BoxEntity
from src.infra.train.checkpoints import S3CheckpointStorageCallback
from src.infra.train.checkpoints import LocalCheckpointStorageCallback

from .inference_callback import InferenceCallback


class PokemonTrainer:

    def __init__(
        self,
        context_length=4096,
        row_length=64,
    ):

        self.row_length = row_length
        self.context_length = context_length

    def train(
        self,
        box_entity: BoxEntity,
    ):

        name = box_entity.name
        dataset = box_entity.dataset
        tokenizer = box_entity.tokenizer

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer.get_vocab()),
                n_ctx=self.context_length,
                n_positions=self.context_length,
                n_embd=128,  # tamaño del embedding (por defecto GPT2 usa 768)
                n_layer=4,  # número de capas Transformer (por defecto 6)
                n_head=4,  # número de cabezas de atención (por defecto 12)
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdirname:

            checkpoint_storage_callback = LocalCheckpointStorageCallback(
                dataset_name=name,
            )

            trainer_args = TrainingArguments(
                output_dir=tmpdirname,
                overwrite_output_dir=True,
                per_device_train_batch_size=4,
                num_train_epochs=100,
                logging_steps=10,
                gradient_accumulation_steps=4,
                save_strategy="steps",
                save_steps=1,
                learning_rate=5e-4,
                weight_decay=0.0,
                warmup_ratio=0.05,
                fp16=torch.cuda.is_available(),
                dataloader_pin_memory=torch.cuda.is_available(),
            )

            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                args=trainer_args,
                data_collator=data_collator,
                train_dataset=dataset["train"]["input_ids"],
                callbacks=[
                    InferenceCallback(
                        tokenizer,
                        interval_steps=50,
                        row_length=self.row_length,
                        context_length=4096,
                    ),
                    checkpoint_storage_callback,
                ],
            )

            trainer.train(
                resume_from_checkpoint=checkpoint_storage_callback.resume_from_checkpoint
            )

        return self
