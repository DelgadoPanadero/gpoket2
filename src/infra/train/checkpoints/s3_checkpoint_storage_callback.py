import os
import re

import boto3
from transformers import TrainerState  # type: ignore
from transformers import TrainerControl  # type: ignore
from transformers import TrainerCallback  # type: ignore
from transformers import TrainingArguments  # type: ignore
from botocore.exceptions import ClientError


class S3CheckpointStorageCallback(TrainerCallback):

    def __init__(
        self,
        dataset_name: str,
    ):
        self.bucket_name = "model"
        self.prefix = dataset_name
        self._previous_last_step = 0
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
        )

    def _list_checkpoints(
        self,
    )->list:
        try:
            resp = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.prefix}/checkpoint-",
            )

            checkpoints = []
            for obj in resp.get("Contents",[]):
                if match := re.search(r"checkpoint-(\d+)", obj["Key"]):
                    checkpoints.append(
                        obj["Key"].split("/")[0] + "/" + match.group(0)
                        )

            return list(set(checkpoints))

        except ClientError:
            return []

    def _download_checkpoint(
        self,
        checkpoint_prefix: str,
        output_dir: str,
    ):
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket_name, Prefix=checkpoint_prefix
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                checkpoint_name = checkpoint_prefix.split("/")[-1]
                rel_path = key[len(checkpoint_prefix) :].lstrip("/")
                dest_path = os.path.join(output_dir, checkpoint_name, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                self.s3_client.download_file(self.bucket_name, key, dest_path)


    def _upload_checkpoint(
        self,
        local_checkpoint_path: str,
        step:int,
    ):
        checkpoint_prefix = f"{self.prefix}/checkpoint-{step}"
        for root, _, files in os.walk(local_checkpoint_path):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_checkpoint_path)
                s3_key = f"{checkpoint_prefix}/{rel_path}"
                self.s3_client.upload_file(local_path, self.bucket_name, s3_key)


    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoints = self._list_checkpoints()
        if checkpoints and args.output_dir:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            last_checkpoint_prefix = checkpoints[-1]

            self._download_checkpoint(last_checkpoint_prefix, args.output_dir)
            self._previous_last_step = int(last_checkpoint_prefix.split("-")[-1])

            args.resume_from_checkpoint = args.output_dir


    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero and args.output_dir:
            step = state.global_step + self._previous_last_step
            checkpoint_dir = f"checkpoint-{state.global_step}"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_dir)
            self._upload_checkpoint(checkpoint_path, step)