import os
import tempfile

import boto3
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast  # type: ignore
from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.gld.prof_oak_pc import ProfOakPcRepository


class S3ProfOakPcRepository(ProfOakPcRepository):

    def __init__(
        self,
        bucket : str = "gld",
        prefix : str = "prof_oak_pc",
        partition : str = ""
            
    ):
        self.bucket = "gld"
        self.prefix = f"{prefix}{partition}".strip("/")

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        )

    def _upload_directory(
        self,
        s3_prefix,
        local_dir,
    ):
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                self.s3_client.upload_file(
                    local_path,
                    self.bucket,
                    f"{s3_prefix}/{relative_path}",
                )

    def _download_directory(
        self,
        s3_prefix,
        local_dir,
    ):
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                s3_path = obj["Key"]
                rel_path = os.path.relpath(s3_path, s3_prefix)
                local_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.s3_client.download_file(
                    self.bucket,
                    s3_path,
                    local_path,
                )

    def save(
        self,
        box_entity: BoxEntity,
    )->str:
        
        box_name = ""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save dataset
            box_entity.dataset.save_to_disk(tmpdir)
            # Save tokenizer
            box_entity.tokenizer.save_pretrained(tmpdir)
            # Upload everything to S3
            s3_prefix = f"{self.prefix}/{box_entity.name}"
            self._upload_directory(tmpdir, s3_prefix)

            box_name = box_entity.name
        
        return box_name
        

    def load(
        self,
        box_name: str | None = None,
    ) -> BoxEntity:

        # List available boxes
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket, Prefix=self.prefix + "/", Delimiter="/"
        )
        box_names = [
            prefix["Prefix"].split("/")[-2]
            for prefix in response.get("CommonPrefixes", [])
            if prefix["Prefix"].startswith(f"{self.prefix}/box-")
        ]
        if not box_names:
            raise FileNotFoundError("No box found in S3 storage.")
        if box_name is None:
            box_name = sorted(box_names, reverse=True)[0]

        s3_prefix = f"{self.prefix}/{box_name}"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._download_directory(s3_prefix, tmpdir)
            dataset = DatasetDict.load_from_disk(tmpdir)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tmpdir)
            return BoxEntity(
                name=box_name,
                dataset=dataset,
                tokenizer=tokenizer,
            )
