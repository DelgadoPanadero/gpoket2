from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.slv.pokedex import PokedexRepository
from src.domain.gld.prof_oak_pc import ProfOakPcRepository
from src.application.gld.augmentation import ColorShift, HorizontalFlip
from src.application.gld.prof_oak_pc.tokenizer import Pokenizer


def get_prof_oak_pc(
    pokedex_repository: PokedexRepository,
    profoakpc_repository: ProfOakPcRepository,
) -> list[str]:
    pokedex_list = pokedex_repository.load_all()

    flip = HorizontalFlip()
    #color_shift = ColorShift()
    augmented_list = []
    for entity in pokedex_list:
        augmented_list.append(entity)
        augmented_list.append(flip.run(entity))
        #augmented_list.extend(color_shift.run(entity))

    pokenizer = Pokenizer().train(augmented_list)
    dataset = pokenizer.tokenize(augmented_list)

    box_entity = BoxEntity(
        name="box-" + profoakpc_repository.partition,
        tokenizer=pokenizer.tokenizer,
        dataset=dataset,
    )

    box_name = profoakpc_repository.save(box_entity)

    return [box_name]
