from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.slv.pokedex import PokedexRepository
from src.domain.gld.prof_oak_pc import ProfOakPcRepository
from src.application.gld.prof_oak_pc.tokenizer import Pokenizer


def get_prof_oak_pc(
    pokedex_repository: PokedexRepository,
    profoakpc_repository: ProfOakPcRepository,
) -> list[str]:

    pokedex_list = pokedex_repository.load_all()

    pokenizer = Pokenizer().train(pokedex_list)
    dataset = pokenizer.tokenize(pokedex_list)

    box_entity = BoxEntity(
        name="box-"+profoakpc_repository.partition,
        tokenizer=pokenizer.tokenizer,
        dataset=dataset,
    )

    box_name = profoakpc_repository.save(box_entity)

    return [box_name]