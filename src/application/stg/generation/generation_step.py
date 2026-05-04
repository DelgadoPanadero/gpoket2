from src.application.stg.generation import GenerationAdapter


class GenerationStep:
    def __init__(
        self,
        adapter: GenerationAdapter,
        generations: list[int] = [1, 2, 3, 4],
    ):
        self.adapter = adapter
        self.generations = generations

    def run(self) -> list[str]:
        return self.adapter.save_all(self.generations)
