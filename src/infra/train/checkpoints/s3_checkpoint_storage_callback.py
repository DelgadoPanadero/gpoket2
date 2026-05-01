import os
import re
from pathlib import Path

import boto3
from transformers import TrainerState  # type: ignore
from transformers import TrainerControl  # type: ignore
from transformers import TrainingArguments  # type: ignore
from botocore.exceptions import ClientError

from src.application.train.checkpoint_storage_callback import (
    CheckpointStorageCallback,
)


class S3CheckpointStorageCallback(CheckpointStorageCallback):
    def __init__(
        self,
        box_name: str = "",
    ):
        self.bucket_name = "train"
        self.prefix = box_name
        self.resume_from_checkpoint = None
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
        )

        if box_name is None:
            all_train_prefix = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Delimiter="/",
            )

            if prefixes := sorted(
                [
                    prefix["Prefix"].rstrip("/")
                    for prefix in all_train_prefix.get("CommonPrefixes", [])
                ],
            ):
                self.prefix = prefixes[-1]

    def _get_latest_checkpoint(
        self,
    ) -> str | None:
        try:
            resp = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.prefix}/checkpoint-",
            )

            checkpoints = []
            for obj in resp.get("Contents", []):
                if match := re.search(r"checkpoint-(\d+)", obj["Key"]):
                    checkpoints.append(
                        {
                            "step": int(match.group(1)),
                            "checkpoint_path": os.path.dirname(obj["Key"]),
                        },
                    )

            last_checkpoint_prefix = None
            if checkpoints:
                checkpoints.sort(key=lambda x: x["step"])
                last_checkpoint_prefix = checkpoints[-1]["checkpoint_path"]

            return last_checkpoint_prefix

        except ClientError:
            return None

    def _load_checkpoint(
        self,
        checkpoint_path: str,
        trainer_checkpoint_dir: str,
    ):
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=checkpoint_path,
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                checkpoint_name = checkpoint_path.split("/")[-1]
                rel_path = key[len(checkpoint_path) :].lstrip("/")
                dest_path = os.path.join(
                    trainer_checkpoint_dir,
                    checkpoint_name,
                    rel_path,
                )
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                self.s3_client.download_file(
                    Bucket=self.bucket_name,
                    Key=key,
                    Filename=dest_path,
                )

    def _save_checkpoint(
        self,
        trainer_checkpoint_path: str,
    ):
        checkpoint_name = os.path.basename(trainer_checkpoint_path)
        checkpoint_prefix = f"{self.prefix}/{checkpoint_name}"
        for root, _, files in os.walk(trainer_checkpoint_path):
            for file in files:
                trainer_path = os.path.join(root, file)
                rel_path = os.path.relpath(
                    trainer_path,
                    trainer_checkpoint_path,
                )
                s3_key = f"{checkpoint_prefix}/{rel_path}"
                self.s3_client.upload_file(
                    Filename=trainer_path,
                    Bucket=self.bucket_name,
                    Key=s3_key,
                )

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        last_checkpoint_path = self._get_latest_checkpoint()
        if last_checkpoint_path and args.output_dir:
            self._load_checkpoint(
                checkpoint_path=last_checkpoint_path,
                trainer_checkpoint_dir=args.output_dir,
            )

            self.resume_from_checkpoint = os.path.join(
                args.output_dir,
                os.path.basename(last_checkpoint_path),
            )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero and args.output_dir:
            self._save_checkpoint(
                os.path.join(
                    args.output_dir,
                    f"checkpoint-{state.global_step}",
                ),
            )
