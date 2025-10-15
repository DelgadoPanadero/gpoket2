import os
from pathlib import Path

from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast  # type: ignore
from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.gld.prof_oak_pc import ProfOakPcRepository


class LocalProfOakPcRepository(ProfOakPcRepository):
    def __init__(
        self,
        base_dir: Path | str = Path("/home/data/gld"),
        entity: str = "prof_oak_pc",
        partition: str = "latest",
    ):
        self.partition = partition
        self.base_dir = Path(base_dir) / Path(entity)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        box_entity: BoxEntity,
    ) -> str:
        source_dir = self.base_dir / Path(box_entity.name)
        source_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset
        box_entity.dataset.save_to_disk(str(source_dir))

        # Save tokenizer
        box_entity.tokenizer.save_pretrained(str(source_dir))

        return box_entity.name

    def load(
        self,
    ) -> BoxEntity:
        if self.partition != "latest":
            box_entity_name = "box-latest"

        else:
            if all_partitions := sorted(
                [
                    dir_name.name
                    for dir_name in self.base_dir.glob("box-*")
                    if dir_name.is_dir()
                ],
            ):
                box_entity_name = all_partitions[-1]

        box_dir_path = Path(self.base_dir) / Path(box_entity_name)
        dataset = DatasetDict.load_from_disk(str(box_dir_path))

        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(box_dir_path))

        return BoxEntity(
            name=self.base_dir.name,
            dataset=dataset,
            tokenizer=tokenizer,
        )
