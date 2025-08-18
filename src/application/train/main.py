from src.domain.gld.prof_oak_pc import ProfOakPcRepository
from src.application.train.pokemon_trainer import PokemonTrainer


def train_pokemons(
    profoakpc_repository: ProfOakPcRepository,
):

    box_entity = profoakpc_repository.load()

    PokemonTrainer(box_entity).create_trainer().train()


if __name__ == "__main__":

    from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository

    train_pokemons(
        profoakpc_repository=LocalProfOakPcRepository(),
    )
