from pydantic import BaseModel
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast #type: ignore


class BoxEntity(BaseModel):
    name: str
    dataset: DatasetDict
    tokenizer: PreTrainedTokenizerFast

    model_config = {"arbitrary_types_allowed": True}
