from typing import Protocol, runtime_checkable


@runtime_checkable
class PokenizerProtocol(Protocol):
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    def get_vocab(self) -> dict: ...
    def convert_tokens_to_ids(self, token: str) -> int: ...
    def save_pretrained(self, path: str) -> None: ...
