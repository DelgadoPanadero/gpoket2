from src.encoder import Pokedex
from src.dataset import PokeCenter
from src.tokenizer import Pokenizer
from src.model_trainer import PokemonTrainer


MODEL_DIR = "./model"
IMAGES_DIR = "./pokemons"
DATASET_DIR = "./pokedex"
TOKENIZER_FILE = "./model/tokenizer.json"

ROW_LENGTH = 22
CTX_LENGTH = 506


if __name__ == "__main__":

    Pokedex().batch_files_encoding(
        IMAGES_DIR,
        DATASET_DIR,
    )

    Pokenizer(
        ROW_LENGTH,
        CTX_LENGTH,
    ).train(
        DATASET_DIR,
    ).save(
        MODEL_DIR,
    )

    PokeCenter(
        MODEL_DIR,
        ROW_LENGTH,
        CTX_LENGTH,
    ).create_dataset(
        DATASET_DIR
    )

    PokemonTrainer(
        DATASET_DIR,
        MODEL_DIR,
        ROW_LENGTH,
        CTX_LENGTH,
    ).create_trainer().train()
