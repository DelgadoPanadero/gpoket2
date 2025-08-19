import os
import json
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast  # type: ignore
from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.gld.prof_oak_pc import ProfOakPcRepository


class LocalProfOakPcRepository(ProfOakPcRepository):

    source_dir = f"/home/data/gld/prof_oak_pc"

    def save(
        self,
        box_entity: BoxEntity,
    ):

        source_dir = f"{self.source_dir}/{box_entity.name}"
        os.makedirs(source_dir, exist_ok=True)

        # Save dataset
        box_entity.dataset.save_to_disk(source_dir)

        # Save tokenizer
        box_entity.tokenizer.save_pretrained(source_dir)

    def load(
        self,
        box_name: str | None = None,
    ) -> BoxEntity:

        if box_name is None:
            box_name = sorted(
                [
                    dir_name
                    for dir_name in os.listdir(self.source_dir)
                    if dir_name.startswith("box-")
                ],
                reverse=True,
            )[0]

        source_dir = f"{self.source_dir}/{box_name}"

        dataset = DatasetDict.load_from_disk(source_dir)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(source_dir)

        return BoxEntity(
            name=box_name,
            dataset=dataset,
            tokenizer=tokenizer,
        )
