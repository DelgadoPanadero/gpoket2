from abc import ABC, abstractmethod
from src.domain.gld.prof_oak_pc import BoxEntity


class ProfOakPcRepository(ABC):

    @abstractmethod
    def save(
        self,
        box_entity: BoxEntity,
    ):

        NotImplementedError()

    @abstractmethod
    def load(
        self,
        box_name: str | None = None,
    ) -> BoxEntity:

        NotImplementedError()
