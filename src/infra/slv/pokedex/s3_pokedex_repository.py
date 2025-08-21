import os
import boto3
from pathlib import Path
from src.domain.slv.pokedex import PokedexEntity
from src.domain.slv.pokedex import PokedexRepository


class S3PokedexRepository(PokedexRepository):

    def __init__(
        self,
        bucket: str = "slv",
        entity: str = "pokedex",
        partition: str = "",
    ):

        self.bucket = bucket
        self.prefix = f"{entity}/{partition}".strip("/")

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
        )

    def load_one(
        self,
        pokedex_item_path: str,
    ) -> PokedexEntity:

        obj = self.s3_client.get_object(
            Bucket=self.bucket,
            Key=pokedex_item_path,
        )

        data = obj["Body"].read().decode("utf-8")

        return PokedexEntity(
            name=Path(pokedex_item_path).name,
            data=data,
        )

    def save_one(
        self,
        pokedex_item: PokedexEntity,
        pokedex_item_path: str,
    ) -> str:

        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=pokedex_item_path,
            Body=pokedex_item.data.encode("utf-8"),
        )

        return pokedex_item_path

    def save_all(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> list[str]:

        pokedex_item_path_list = []
        for pokedex_item in pokedex_list:
            if pokedex_item_path := self.save_one(
                pokedex_item=pokedex_item,
                pokedex_item_path=f"{self.prefix}/{pokedex_item.name}",
            ):
                pokedex_item_path_list.append(pokedex_item_path)

        return pokedex_item_path_list

    def load_all(
        self,
    ) -> list[PokedexEntity]:

        pokedex_entity_list = []

        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".txt"):
                    if pokedex_entity := self.load_one(obj["Key"]):
                        pokedex_entity_list.append(pokedex_entity)

        return pokedex_entity_list
