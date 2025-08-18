import os

import torch
from datasets import load_from_disk
from transformers import AutoConfig
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling

from transformers import Trainer
from transformers import TrainingArguments

from .inference_callback import InferenceCallback


class PokemonTrainer:

    __base_model__ = "distilgpt2"

    def __init__(
        self,
        dataset_dir,
        model_dir,
        row_length=32,
        context_length=1024,
    ):

        self._model_dir = model_dir
        self._row_length = row_length
        self._context_length = context_length

        self._dataset = load_from_disk(dataset_dir)
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        self._tokenizer.add_special_tokens(
            {
                'pad_token': '[PAD]',
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
            },
        )

        self._data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        self._model = GPT2LMHeadModel(
            AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.__base_model__,
                vocab_size=len(self._tokenizer),
                n_ctx=self._context_length,
                bos_token_id=self._tokenizer.bos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                loss_type="ForCausalLMLoss",
            )
        )

    def create_trainer(self, **kwargs):
        """ """

        default_args = {
            "output_dir": self._model_dir,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "logging_steps": 5_000,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
            "weight_decay": 0.1,
            "warmup_steps": 1_000,
            "lr_scheduler_type": "cosine",
            "learning_rate": 5e-4,
            "save_steps": 5_000,
            "fp16": torch.cuda.is_available(),
            "dataloader_pin_memory": torch.cuda.is_available(),
        }

        default_args.update(kwargs)
        trainer_args = TrainingArguments(**default_args)

        trainer = Trainer(
            model=self._model,
            tokenizer=self._tokenizer,
            args=trainer_args,
            data_collator=self._data_collator,
            train_dataset=self._dataset["train"],
            callbacks=[InferenceCallback(self._tokenizer, interval_steps=10)],
        )

        return trainer
