import os
import cv2
import boto3
import requests
import numpy as np
from io import BytesIO
from pathlib import Path

from src.domain.brz.pokemon import PokemonEntity
from src.domain.brz.pokemon import PokemonRepository


class S3PokemonRepository(PokemonRepository):

    def __init__(self, bucket: str = "brz", prefix: str = "pokemons"):

        self.bucket = bucket
        self.prefix = prefix
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
        )

        self.api_url = (
            "https://api.github.com/repos/DelgadoPanadero/GPokeT2"
            "/contents/data/bzr/pokemons?ref=main"
        )

    def load_one(
        self,
        img_path: str,
    ) -> PokemonEntity:

        response = self.s3_client.get_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/{img_path}",
        )

        image_bytes = response["Body"].read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        response["Body"].close()

        return PokemonEntity(
            name=img_path,
            image=image,
        )

    def load_all(
        self,
        partition_name: str = "",
    ) -> list[PokemonEntity]:

        result = []

        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=f"{self.prefix}/{partition_name}",
        ):

            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".png"):
                    result.append(self.load_one(Path(obj["Key"]).name))

        return result

    def save_one(
        self,
        file_info: dict,
    ) -> str:

        object_name = ""
        if file_info["type"] == "file" and file_info["name"].endswith(".png"):
            response = requests.get(file_info["download_url"])
            if response.status_code == 200:
                key = f"{self.prefix}{file_info['name']}"
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=BytesIO(response.content),
                    ContentType="image/png",
                )
                object_name = key
        return object_name

    def save_all(
        self,
        partition_name: str = "",
    ) -> str:
        response = requests.get(self.api_url)
        if response.status_code != 200:
            raise Exception(
                f"Error al acceder a {self.api_url}: {response.status_code}"
            )
        files = response.json()
        object_name_list = []
        for file_info in files:
            object_name = self.save_one(file_info)
            object_name_list += [object_name] if object_name else []

        return partition_name
