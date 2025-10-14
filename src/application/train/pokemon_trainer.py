import tempfile

import torch
from transformers import Trainer  # type: ignore
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore

from src.domain.gld.prof_oak_pc import BoxEntity
from .inference_callback import InferenceCallback
from .checkpoint_storage_callback import CheckpointStorageCallback


class WeightedTokenLossTrainer(Trainer):
    def __init__(
        self,
        *args,
        weight_token_id=None,
        token_weight=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_token_id = weight_token_id
        self.token_weight = token_weight

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        vocab_size = logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))

        # Ajuste de peso para token "~"
        if self.weight_token_id is not None:
            weights = torch.ones_like(labels.view(-1), dtype=loss.dtype, device=loss.device)
            weights[labels.view(-1) == self.weight_token_id] = self.token_weight
            loss = loss * weights

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


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

        self.inference_callback = InferenceCallback(
            context_length=4096,
            row_length=64,
            interval_steps=10,
            tokenizer=tokenizer,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer.get_vocab()),
                n_ctx=self.context_length,
                n_positions=self.context_length,
                n_embd=72,  # tamaño del embedding (por defecto GPT2 usa 768)
                n_layer=6,  # número de capas Transformer (por defecto 6)
                n_head=12,  # número de cabezas de atención (por defecto 12)
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer_args = TrainingArguments(
                output_dir=tmpdirname,
                overwrite_output_dir=True,
                per_device_train_batch_size=64,
                num_train_epochs=3000,
                logging_steps=10,
                gradient_accumulation_steps=16,
                save_strategy="steps",
                save_steps=50,
                learning_rate=0.1e-7,
                weight_decay=0.0,
                warmup_ratio=0.05,
                fp16=torch.cuda.is_available(),
                dataloader_pin_memory=torch.cuda.is_available(),
            )

            trainer = WeightedTokenLossTrainer(
                model=model,
                processing_class=tokenizer,
                args=trainer_args,
                data_collator=data_collator,
                train_dataset=dataset["train"],
                weight_token_id=tokenizer.convert_tokens_to_ids("~"),
                token_weight=0.1,
                callbacks=[
                    self.inference_callback,
                    self.checkpoint_storage_callback,
                ],
            )

            trainer.train(
                resume_from_checkpoint=self.checkpoint_storage_callback.resume_from_checkpoint
            )

        return self
