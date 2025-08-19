import os
import boto3
from pathlib import Path
from src.domain.slv.pokedex import PokedexEntity
from src.domain.slv.pokedex import PokedexRepository


class S3PokedexRepository(PokedexRepository):

    bucket = "slv"
    prefix = "pokedex"
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
    )

    def load_one(
        self,
        img_path: Path,
    ) -> PokedexEntity:

        obj = self.s3_client.get_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/{img_path.name}",
        )

        data = obj["Body"].read().decode("utf-8")

        return PokedexEntity(
            name=img_path.name,
            data=data,
        )

    def save_one(
        self,
        pokedex_item: PokedexEntity,
    ) -> None:

        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/{pokedex_item.name}",
            Body=pokedex_item.data.encode("utf-8"),
        )

    def save_all(
        self,
        pokedex_list: list[PokedexEntity],
    ) -> None:
        for pokedex_item in pokedex_list:
            self.save_one(pokedex_item)

    def load_all(
        self,
    ) -> list[PokedexEntity]:
        paginator = self.s3_client.get_paginator("list_objects_v2")
        result = []
        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=self.prefix + "/"
        ):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".txt"):
                    name = Path(obj["Key"]).name
                    pokedex_entity = self.load_one(Path(name))
                    result.append(pokedex_entity)
        return result
