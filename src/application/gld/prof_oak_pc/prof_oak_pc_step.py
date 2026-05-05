from tqdm import tqdm

from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.slv.pokedex import PokedexEntity
from src.domain.slv.pokedex import PokedexRepository
from src.domain.gld.prof_oak_pc import ProfOakPcRepository
from src.application.gld.prof_oak_pc.filter import SizeFilter
from src.application.gld.prof_oak_pc.augmentation import ColorShift
from src.application.gld.prof_oak_pc.augmentation import HorizontalFlip
from src.application.gld.prof_oak_pc.tokenizer import Pokenizer


class ProfOakPcStep:
    def __init__(
        self,
        pokedex_repository: PokedexRepository,
        profoakpc_repository: ProfOakPcRepository,
        context_length: int = 1024,
    ):
        self.pokedex_repository = pokedex_repository
        self.profoakpc_repository = profoakpc_repository
        self.context_length = context_length

    def run(self) -> list[str]:
        pokedex_list = self.pokedex_repository.load_all()

        flip = HorizontalFlip()
        # color_shift = ColorShift()
        size_filter = SizeFilter(size=64)

        augmented_list = []
        for entity in tqdm(pokedex_list, desc="Processing Pokémon entities"):
            if entity := size_filter.run(entity):
                augmented_list.append(entity)
                augmented_list.append(flip.run(entity))
                # augmented_list.extend(color_shift.run(entity))

        pokenizer = Pokenizer(context_length=self.context_length).train(
            augmented_list
        )
        dataset = pokenizer.tokenize(augmented_list)

        meta = {
            p.name: {
                f: getattr(p, f)
                for f in [
                    f
                    for f in PokedexEntity.model_fields
                    if f not in ("name", "data")
                ]
            }
            for p in augmented_list
        }

        dataset = dataset.map(lambda row: meta[row["name"]])

        box_entity = BoxEntity(
            name="box-" + self.profoakpc_repository.partition,
            tokenizer=pokenizer.tokenizer,
            dataset=dataset,
        )

        box_name = self.profoakpc_repository.save(box_entity)

        return [box_name]
