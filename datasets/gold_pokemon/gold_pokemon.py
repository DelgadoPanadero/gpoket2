import datasets
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import WhitespaceSplit

class GoldPokemon(datasets.GeneratorBasedBuilder):
    """Gold layer: Tokenizes Pokemon text data for training."""

    # Tokenizer constants (same as in Pokenizer)
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    PAD_TOKEN = "[PAD]"
    BOL_TOKEN = "00"
    BCK_TOKEN = "~"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_length = 64
        self.col_length = 64
        self.context_length = 4096
        self.chunk_step_rows = 1
        self.tokenizer = None

    def _info(self):
        return datasets.DatasetInfo(
            description="Tokenized Pokemon data ready for training.",
            features=datasets.Features({
                'name': datasets.Value("string"),
                'chunk': datasets.Value("int64"),
                'labels': datasets.Sequence(datasets.Value("int64")),
                'input_ids': datasets.Sequence(datasets.Value("int64")),
                'input_text': datasets.Value("string"),
                'original_text': datasets.Value("string"),
                'attention_mask': datasets.Sequence(datasets.Value("int64")),
            }),
        )

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

    def _generate_examples(self, **kwargs):
        """Loads the Silver dataset, trains tokenizer, and yields tokenized examples."""
        # Load the silver layer dataset
        silver_ds = datasets.load_dataset(path="../silver_pokemon", split="train", streaming=True)
        
        # Collect all text data for training the tokenizer
        text_data = []
        silver_examples = []
        
        for example in silver_ds:
            cleaned_text = self._clean_text(example['encoded_data'])
            text_data.append(cleaned_text)
            silver_examples.append({
                'name': example['name'],
                'text': cleaned_text,
                'original_text': example['encoded_data']
            })
        
        # Train the tokenizer
        self._train_tokenizer(text_data)
        
        # Tokenize and yield examples
        key = 0
        for example in silver_examples:
            # Tokenize the text and create chunks
            text_chunked = self._chunk_text(example['text'].split())
            
            for i, chunk in enumerate(text_chunked):
                chunk_text = " ".join(chunk)
                
                # Tokenize the chunk
                tokenized = self.tokenizer(chunk_text, return_tensors=None)
                
                yield key, {
                    'name': example['name'],
                    'chunk': i + 1,
                    'labels': tokenized["input_ids"],
                    'input_ids': tokenized["input_ids"],
                    'input_text': chunk_text,
                    'original_text': example['original_text'],
                    'attention_mask': tokenized["attention_mask"],
                }
                key += 1

    def _clean_text(self, text: str) -> str:
        """Clean text by adding BOL tokens (same as Pokenizer._clean_text)."""
        text_split = text.split("\n")
        text_split = [[self.BOL_TOKEN] + row.split()[1:] for row in text_split]
        return " ".join([" ".join(row) for row in text_split])

    def _train_tokenizer(self, text_data: List[str]):
        """Train the tokenizer (same logic as Pokenizer.train)."""
        # Initialize tokenizer
        _tokenizer = Tokenizer(WordLevel())
        _tokenizer.pre_tokenizer = WhitespaceSplit()

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=_tokenizer,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            pad_token=self.PAD_TOKEN,
        )

        # Calculate vocab size
        vocab_size = len(self.tokenizer.vocab)
        vocab_size += len(set("".join(text_data)))

        # Train the tokenizer
        self.tokenizer = self.tokenizer.train_new_from_iterator(
            text_iterator=text_data,
            vocab_size=vocab_size,
        )

    def _chunk_text(self, text_split: List[str]) -> List[List[str]]:
        """Chunk text into context-length pieces (same as Pokenizer._chunk_text)."""
        text_split_chunked = []
        step = self.chunk_step_rows * self.row_length
        text_split_padded = text_split + [self.PAD_TOKEN] * self.context_length

        for i in range(0, len(text_split) - self.context_length + 1, step):
            text_split_chunked.append(
                text_split_padded[i : i + self.context_length],
            )

        return text_split_chunked
