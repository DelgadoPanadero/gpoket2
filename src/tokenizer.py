import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit


class Pokenizer(object):

    def __init__(
        self,
        row_length=32,
        context_length=1024,
    ):
        """ """

        self._row_length = row_length
        self._context_length = context_length

        self._tokenizer = Tokenizer(BPE(
            vocab=self.get_vocab(),
            merges=[],
        ))
        self._tokenizer.normalizer = NFKC()
        self._tokenizer.pre_tokenizer = WhitespaceSplit()

    def get_vocab(self):
        """ """
        vocab = ["~","[BOS]", "[EOS]"]
        vocab += ["%02d" % i for i in range(self._row_length)]
        vocab += [chr(i + 59) for i in range(64)]
        vocab = {w:i for i,w in enumerate(vocab)}
        return vocab

    def train(self, images_dir):
        """ """

        paths = [str(x) for x in Path("./pokemons_txt").glob("**/*.txt")]

        vocab = self.get_vocab()
        trainer = BpeTrainer(
            vocab_size=len(vocab)+5,
            show_progress=True,
            special_tokens = ["[BOS]", "[EOS]"],
        )

        self._tokenizer.train(files=paths, trainer=trainer)

        return self

    def save(
        self,
        model_dir,
        prefix=None,
    ):
        """ """

        os.makedirs(model_dir, exist_ok=True)
        file_name = os.path.join(model_dir, "tokenizer.json")
        self._tokenizer.save(file_name)
        self._tokenizer.model.save(model_dir, prefix)
