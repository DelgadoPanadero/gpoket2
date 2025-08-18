import os
import torch
from transformers import Trainer  # type: ignore
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from .inference_callback import InferenceCallback
from src.domain.gld.prof_oak_pc import BoxEntity


class WeightedLossTrainer(Trainer):

    def __init__(
        self,
        *args,
        loss_weights=None,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights

    def compute_loss(  # type: ignore
        self,
        model,
        inputs,
        return_outputs=False,
        *,
        num_items_in_batch=None,
    ):

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.loss_weights.to(logits.device),  # type: ignore
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class PokemonTrainer:

    def __init__(
        self,
        box_entity: BoxEntity,
        context_length=4096,
        row_length=64,
    ):

        self.row_length = row_length
        self.context_length = context_length


        self.dataset = box_entity.dataset
        self.tokenizer = box_entity.tokenizer
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(self.tokenizer.get_vocab()),
                n_ctx=self.context_length,
                n_positions=self.context_length,
                n_embd=128,  # tamaño del embedding (por defecto GPT2 usa 768)
                n_layer=4,  # número de capas Transformer (por defecto 6)
                n_head=4,  # número de cabezas de atención (por defecto 12)
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        )

    def create_trainer(self, **kwargs):
        model_dir = "/home/data/model"
        os.makedirs(model_dir, exist_ok=True)

        default_args = {
            "output_dir": model_dir,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": 1,
            "num_train_epochs": 200,
            "logging_steps": 10,
            "gradient_accumulation_steps": 1,
            "save_strategy": "epoch",
            "learning_rate": 5e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.05,
            "fp16": torch.cuda.is_available(),
            "dataloader_pin_memory": torch.cuda.is_available(),
        }

        default_args.update(kwargs)
        trainer_args = TrainingArguments(**default_args)

        # Crear vector de pesos para la pérdida
        weights = torch.ones(len(self.tokenizer))
        weights[self.tokenizer.convert_tokens_to_ids("~")] = 0.1
        weights[self.tokenizer.convert_tokens_to_ids("00")] = 10  # penalizar menos el token "~"

        trainer = Trainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=trainer_args,
            data_collator=self.data_collator,
            train_dataset=self.dataset["train"]["input_ids"],
            callbacks=[
                InferenceCallback(
                    self.tokenizer,
                    interval_steps=50,
                    row_length=self.row_length,
                    context_length=4096,
                )
            ],
            #loss_weights=weights,

        )

        return trainer
