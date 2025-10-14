from transformers import TrainerCallback  # type: ignore
from src.domain.gld.prof_oak_pc import ProfOakPcRepository
from src.application.train.pokemon_trainer import PokemonTrainer


def train_pokemons(
    profoakpc_repository: ProfOakPcRepository,
    checkpoint_storage_callback: TrainerCallback | None = None,
):
    box_entity = profoakpc_repository.load()

    PokemonTrainer(
        checkpoint_storage_callback=checkpoint_storage_callback,
    ).train(box_entity)
