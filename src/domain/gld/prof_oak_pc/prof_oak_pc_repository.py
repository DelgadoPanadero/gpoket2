from abc import ABC, abstractmethod
from src.domain.gld.prof_oak_pc import BoxEntity


class ProfOakPcRepository(ABC):

    partition = ""

    @abstractmethod
    def save(
        self,
        box_entity: BoxEntity,
    ) -> str:

        NotImplementedError()

    @abstractmethod
    def load(
        self,
    ) -> BoxEntity:

        NotImplementedError()
