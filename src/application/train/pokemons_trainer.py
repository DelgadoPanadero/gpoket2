import os
import json
import torch
from tokenizers import Tokenizer
from transformers import Trainer # type: ignore
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments # type: ignore
from transformers import PreTrainedTokenizerFast # type: ignore
from transformers import DataCollatorForLanguageModeling # type: ignore
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
        context_length=1024,
        row_length=64,
    ):

        self._row_length = row_length
        self._context_length = context_length

        tokenizer_object = Tokenizer.from_str(
            json.dumps(box_entity.tokenizer, ensure_ascii=False),
        )

        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object,
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token=None,
        )

        self._dataset = box_entity.dataset

        self._data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )


        self._model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(self._tokenizer.get_vocab()),
                n_ctx=self._context_length,
                n_positions=self._context_length,
                n_embd=64,                                                      # tamaño del embedding (por defecto GPT2 usa 768)
                n_layer=4,                                                      # número de capas Transformer (por defecto 6)
                n_head=4,                                                       # número de cabezas de atención (por defecto 12)
                bos_token_id=self._tokenizer.bos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        )

    def create_trainer(self, **kwargs):
        model_dir = "/home/data/model"
        os.makedirs(model_dir, exist_ok=True)

        default_args = {
            "output_dir": model_dir,
            "per_device_train_batch_size": 1,
            "logging_steps": 10,
            "gradient_accumulation_steps": 5,
            "num_train_epochs": 50,
            "warmup_steps": 1000,
            "weight_decay": 0.1,
            "lr_scheduler_type": "cosine",
            "learning_rate": 1e-2,
            "save_steps": 100,
            "fp16": torch.cuda.is_available(),
            "dataloader_pin_memory": torch.cuda.is_available(),
        }

        default_args.update(kwargs)
        trainer_args = TrainingArguments(**default_args)

        # Crear vector de pesos para la pérdida
        weights = torch.ones(len(self._tokenizer))
        weights[self._tokenizer.convert_tokens_to_ids("~")] = 0.1
        #weights[self._tokenizer.convert_tokens_to_ids("00")] = 10  # penalizar menos el token "~"

        trainer = Trainer(
            model=self._model,
            processing_class=self._tokenizer,
            args=trainer_args,
            data_collator=self._data_collator,
            train_dataset=self._dataset["train"],
            loss_weights=weights,
            callbacks=[
                InferenceCallback(
                    self._tokenizer,
                    interval_steps=100,
                    row_length=self._row_length,
                    context_length=1024,
                )
            ],
        )

        return trainer