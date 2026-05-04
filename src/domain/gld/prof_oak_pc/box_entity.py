from datasets import DatasetDict
from pydantic import BaseModel

from .tokenizer_protocol import PokenizerProtocol


class BoxEntity(BaseModel):
    name: str
    dataset: DatasetDict
    tokenizer: PokenizerProtocol

    model_config = {"arbitrary_types_allowed": True}
