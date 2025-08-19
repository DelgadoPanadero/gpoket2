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

    bucket_name = "brz"
    prefix = "pokemon/"
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
    )

    # Github repo info
    api_url = (
        "https://api.github.com/repos/DelgadoPanadero/GPokeT2"
        "/contents/data/bzr/pokemons?ref=main"
    )

    def load_one(
        self,
        object_name: str,
    ) -> PokemonEntity:

        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=f"{self.prefix}{object_name}",
        )

        image_bytes = response["Body"].read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        response["Body"].close()

        return PokemonEntity(
            name=object_name,
            image=image,
        )

    def load_all(
        self,
    ) -> list[PokemonEntity]:

        paginator = self.s3_client.get_paginator("list_objects_v2")
        result = []
        for page in paginator.paginate(
            Bucket=self.bucket_name, Prefix=self.prefix
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
                    Bucket=self.bucket_name,
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
        for file_info in files:
            object_name = self.save_one(file_info)
            object_name_list += [object_name] if object_name else []

        return object_name_list
