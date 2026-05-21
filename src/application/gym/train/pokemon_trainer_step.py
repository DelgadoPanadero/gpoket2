import torch
from transformers import Trainer
from transformers import GPT2Config
from transformers import TrainingArguments

from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.gld.prof_oak_pc import ProfOakPcRepository
from src.domain.gym.model import ModelCard, ModelRepository
from src.application.gym.model import ConditionedGPT2
from src.application.gym.model import ConditionedDataCollator
from src.application.gym.train.callbacks import CheckpointStorageCallback
from src.application.gym.train.callbacks import InferenceCallback


class PokemonTrainerStep:
    def __init__(
        self,
        profoakpc_repository: ProfOakPcRepository,
        checkpoint_storage_adapter,
        context_length: int = 4096,
        row_length: int = 64,
        output_dir: str = "/workspace/train",
        model_repository: ModelRepository | None = None,
        hf_repo_id: str | None = None,
        hf_version: str | None = None,
    ):
        self.row_length = row_length
        self.context_length = context_length
        self.output_dir = output_dir
        self.profoakpc_repository = profoakpc_repository
        self.checkpoint_storage_adapter = checkpoint_storage_adapter
        self.model_repository = model_repository
        self.hf_repo_id = hf_repo_id
        self.hf_version = hf_version

    def train(self, box_entity: BoxEntity):
        dataset = box_entity.dataset
        tokenizer = box_entity.tokenizer

        # Derive num_pokemon from dataset — avoids coupling to Pokenizer internals
        num_pokemon = int(max(dataset["train"]["pokemon_idx"])) + 1

        # Derive row marker token ids from tokenizer for the row embeddings
        row_marker_token_ids = [
            tokenizer.convert_tokens_to_ids(f"[ROW_{i:02d}]") for i in range(64)
        ]

        vocab = tokenizer.get_vocab()
        id_to_tok = {v: k for k, v in vocab.items()}

        # Downweight pure-background BPE tokens (~) to focus training on color pixels
        token_weights = torch.ones(len(vocab))
        for token_id, token_str in id_to_tok.items():
            if token_str and all(c == "~" for c in token_str):
                token_weights[token_id] = 0.6

        self.inference_callback = InferenceCallback(
            context_length=self.context_length,
            interval_steps=100,
            tokenizer=tokenizer,
        )
        self.checkpoint_storage_callback = CheckpointStorageCallback(
            checkpoint_storage_adapter=self.checkpoint_storage_adapter,
        )
        data_collator = ConditionedDataCollator(tokenizer=tokenizer, mlm=False)

        model = ConditionedGPT2(
            config=GPT2Config(
                vocab_size=len(tokenizer.get_vocab()),
                n_ctx=self.context_length,
                n_positions=self.context_length,
                n_embd=512,
                n_layer=12,
                n_head=8,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                _attn_implementation="sdpa",
            ),
            num_pokemon=num_pokemon,
            noise_std=0.1,
            row_marker_token_ids=row_marker_token_ids,
            token_weights=token_weights,
        )

        trainer_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            num_train_epochs=15,
            logging_steps=50,
            gradient_accumulation_steps=16,
            save_strategy="steps",
            save_steps=200,
            learning_rate=6e-4,
            lr_scheduler_type="cosine",
            weight_decay=0.1,
            warmup_ratio=0.05,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=torch.cuda.is_available(),
            dataloader_num_workers=4,
            optim="adamw_torch_fused",
            torch_compile=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=trainer_args,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            callbacks=[
                self.inference_callback,
                self.checkpoint_storage_callback,
            ],
        )

        trainer.train(
            resume_from_checkpoint=self.checkpoint_storage_callback.resume_from_checkpoint,
        )

        if self.model_repository is not None and self.hf_repo_id is not None:
            checkpoint_path = (
                self.checkpoint_storage_adapter.get_latest_checkpoint()
            )
            if checkpoint_path:
                self.model_repository.upload(
                    checkpoint_path=checkpoint_path,
                    repo_id=self.hf_repo_id,
                    version=self.hf_version,
                    model_card=ModelCard(
                        version=self.hf_version,
                        num_pokemon=num_pokemon,
                        dataset_version=getattr(
                            self.profoakpc_repository,
                            "partition",
                            None,
                        ),
                        context_length=self.context_length,
                        n_embd=512,
                        n_layer=12,
                        n_head=8,
                    ),
                )

        return self

    def run(self):
        box_entity = self.profoakpc_repository.load()
        return self.train(box_entity)
