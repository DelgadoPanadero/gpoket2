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

    def __init__(
        self,
        bucket: str = "brz",
        entity: str = "pokemons",
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
            Key=img_path,
        )

        image_bytes = response["Body"].read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        response["Body"].close()

        return PokemonEntity(
            name=Path(img_path).name,
            image=image,
        )

    def load_all(
        self,
    ) -> list[PokemonEntity]:

        result_list = []

        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".png"):
                    if result := self.load_one(obj["Key"]):
                        result_list.append(result)

        return result_list

    def save_one(
        self,
        file_info: dict,
    ) -> str:

        object_name = ""
        if file_info["type"] == "file" and file_info["name"].endswith(".png"):
            response = requests.get(file_info["download_url"])
            if response.status_code == 200:
                key = f"{self.prefix}/{file_info['name']}"
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
    ) -> list[str]:

        response = requests.get(self.api_url)
        if response.status_code != 200:
            raise Exception(
                f"Error al acceder a {self.api_url}: {response.status_code}"
            )
        files = response.json()

        object_name_list = []
        for file_info in files[0:1]:
            if object_name := self.save_one(file_info):
                object_name_list.append(object_name)

        return object_name_list
