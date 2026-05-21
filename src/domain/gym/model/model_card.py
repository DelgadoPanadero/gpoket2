from dataclasses import dataclass, field


@dataclass
class ModelCard:
    version: str | None = None
    num_pokemon: int | None = None
    dataset_version: str | None = None
    context_length: int | None = None
    n_embd: int | None = None
    n_layer: int | None = None
    n_head: int | None = None
    description: str | None = None
    tags: list[str] = field(
        default_factory=lambda: ["pokemon", "sprite-generation", "gpt2"],
    )
    license: str = "mit"
