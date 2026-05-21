from huggingface_hub import snapshot_download

from src.application.gym.inference import PokemonGenerator
from src.infra.brz.pokemon import LocalPokemonRepository

REPO_ID = "iamthinbaker/GPokeT2"
REVISION = "v0.1-wip-3400"
N_SAMPLES = 1

checkpoint_path = snapshot_download(REPO_ID, revision=REVISION)

generator = PokemonGenerator(
    checkpoint_path=checkpoint_path,
    pokemon_repository=LocalPokemonRepository(
        base_path="data/gld/thinbaker_team"
    ),
    device="cuda",
)

for i in range(N_SAMPLES):
    saved_path, meta = generator.generate()
    print(f"[{i + 1}/{N_SAMPLES}] {meta} -> {saved_path}")
