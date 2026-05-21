from tqdm import tqdm

from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.slv.pokedex import PokedexRepository
from src.domain.gld.prof_oak_pc import ProfOakPcRepository
from src.application.gld.prof_oak_pc.filter import SizeFilter
from src.application.gld.prof_oak_pc.augmentation import ColorShift
from src.application.gld.prof_oak_pc.augmentation import HorizontalFlip
from src.application.gld.prof_oak_pc.tokenizer import Pokenizer
from src.application.gld.prof_oak_pc.metadata_adapter import MetadataAdapter


class ProfOakPcStep:
    def __init__(
        self,
        pokedex_repository: PokedexRepository,
        profoakpc_repository: ProfOakPcRepository,
        context_length: int = 4096,
    ):
        self.pokedex_repository = pokedex_repository
        self.profoakpc_repository = profoakpc_repository
        self.context_length = context_length

    def run(self) -> list[str]:
        pokedex_list = self.pokedex_repository.load_all()

        flip = HorizontalFlip()
        color_shift = ColorShift()
        size_filter = SizeFilter(size=64)

        augmented_list = []
        name_to_color_shift: dict[str, int] = {}

        for entity in tqdm(pokedex_list, desc="Processing Pokémon entities"):
            if entity := size_filter.run(entity):
                # Original (no color shift = index 0)
                augmented_list.append(entity)
                name_to_color_shift[entity.name] = 0
                flipped = flip.run(entity)
                augmented_list.append(flipped)
                name_to_color_shift[flipped.name] = 0

                # Color-shifted versions (indices 1-5)
                for shift_idx, shifted in enumerate(
                    color_shift.run(entity),
                    start=1,
                ):
                    augmented_list.append(shifted)
                    name_to_color_shift[shifted.name] = shift_idx
                    flipped_shifted = flip.run(shifted)
                    augmented_list.append(flipped_shifted)
                    name_to_color_shift[flipped_shifted.name] = shift_idx

        pokenizer = Pokenizer(context_length=self.context_length).train(
            augmented_list,
        )
        dataset = pokenizer.tokenize(augmented_list)

        name_to_generation = {e.name: e.generation for e in augmented_list}
        name_to_game_name = {e.name: e.game_name for e in augmented_list}
        dataset = dataset.map(
            lambda row: {
                "generation": name_to_generation[row["name"]],
                "game_name": name_to_game_name[row["name"]],
                "color_shift": name_to_color_shift[row["name"]],
            },
            desc="Adding generation, game_name and color_shift",
            load_from_cache_file=False,
        )

        metadata_adapter = MetadataAdapter()
        dataset = dataset.map(
            lambda row: metadata_adapter.get_pokemon_metadata(
                row["name"],
                row["generation"],
            ),
            desc="Adding metadata",
            load_from_cache_file=False,
        )

        box_entity = BoxEntity(
            name="box-" + self.profoakpc_repository.partition,
            tokenizer=pokenizer.tokenizer,
            dataset=dataset,
        )

        box_name = self.profoakpc_repository.save(box_entity)

        return [box_name]
