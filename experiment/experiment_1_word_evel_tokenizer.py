# train_ascii_gpt2.py
# pip install transformers tokenizers datasets accelerate torch

import os
import math
import random
from dataclasses import dataclass

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


# 1) ASCII original ------------------------------------------------------------
ASCII_ART = r"""
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R Z Z A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R R ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A Z Z R R ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A Z Z A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A A V Z R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A V V R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R R R V V V R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R R A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R R R V V V V V R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R R R A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ R R R R V V V V V V V R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R R R R R A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ A A V V V V R R R ; ; ; R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R A R R R R R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ A Z Z Z V R R ; ; O O O O ; R R ; ~ R R ~ ~ A A A A A ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R R A O A A R R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ A Z Z R ; ; O O O O O O O ; R R ; R Z Z A A Z Z V V V V A ; ; ; ; ; A A ~ ~ A R A O O O O R R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ A Z V ; O O O O O O O O O O ; A A A V V R Z Z Z R A A R V V Z Z R R R R A ~ A R A O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ A V ; O O O O O O O O O O A A Z Z Z A A A V V ; ; z z A V V V R R R A A ~ ~ A R A O O O O A R R A ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ A V V ; O O _ _ _ _ O O O A Z Z Z Z Z Z V V R A c ; ~ ~ R V R R A ; A ~ ~ ~ ~ A R A O O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ A V R O O _ _ _ _ _ _ O O A R Z Z A V V V V V V V V V V V V R ; ~ ~ ~ ~ ~ ~ ~ A R A O O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ A V V O O _ _ _ _ _ _ _ O O ; V V V V V V V V V V V V V V R R R ; ~ ~ ~ ~ ~ ~ A R R A O O O O O A R R A ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ A V R O _ _ _ _ _ _ _ _ O O ; V ; ; ; ; ; V V V V V R R R R R R R ; ~ ~ ~ ~ ~ A R A O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ A V V O _ _ _ _ _ _ _ _ _ O O O ; z A A A z z ; ; V R A A A R R R R A ~ ~ ~ ~ ~ A R A O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ A V R O _ _ _ _ _ _ _ _ O O O O O ~ ; ; ; ~ A A A ; ; ; ; A A R R R R ; ~ ~ ~ A R R O O O O O O O O A R R A ~ ~ ~ ~ ~ ~ ~
00 ~ A V V O _ _ _ _ _ _ _ _ _ O O O O O O A R A A ; ; A A R R R ; R R R R R R ; ~ A R R A O O O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~
00 ~ A V V O _ _ _ _ _ _ _ _ _ O O O O O A R A ~ ~ ~ ~ ; R R R R A R R A R R R ; A R R A O O O O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~
00 ~ A V R O _ _ _ _ _ c _ _ _ O O O O O A R A ~ ~ ~ ~ A R R R R A R R A R R R R A R A O O O O O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~
00 ~ A V R O _ _ _ _ c _ c _ O O O O O O A R R A ~ ~ ~ R R R R A R R R ; R R R R ; A O O O O O O O _ O O O O R R A ~ ~ ~ ~ ~ ~ ~
00 A V V O _ _ _ _ c _ c _ _ O O O _ O O O A R A A ~ A R R R R ; R R ; R R R R R R ; O O O O O O _ O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~
00 A V V O _ _ _ c c c _ c _ O O _ O _ O O O A R R ; A R R A ; R R ; R R R R R R R A ; O O O O O O _ O O O A R A ~ ~ ~ ~ ~ ~ ~ ~
00 A V V O _ _ c c c c c _ _ O _ O _ O _ O O O R A ~ A A A ~ A V R ; R R R R R R R R ; O O O O O _ _ O O O A R A ~ ~ ~ ~ ~ F ~ ~
00 A V R O _ _ c c c c _ _ _ O O _ O _ O O O O O A V ; ; ; A V R ; ~ A V R R R R R R ; O O O O _ _ O O O O A R A ~ ~ ~ ~ ~ F ~ ~
00 A V R O _ c c c c c c _ O O _ O _ O _ O O O O O A Z V V V R ; ~ ~ A V R R R R R R A ; O O O _ _ O O O O R R ~ ~ ~ ~ ~ ~ F ~ ~
00 V V O _ _ c c ; ; ; c _ O _ O _ O _ O _ O O O O O ; ; ; ; ; ~ ~ ~ A V R R R R R R R ; O O O O O O O O A R A ~ ~ ~ ~ R ~ F R ~
00 V V O _ c c ; ~ ~ ~ O _ O _ _ _ O ; ; ; ; O O O O O O ; R R A ~ ~ A V V R R R R R R ; O O O O O O O O A R A ~ ~ F ~ R ~ F R ~
00 V V O _ c ; ~ ~ ~ ~ ~ O _ _ O O ~ ~ ~ ~ ~ ; ; O O O O O ; R R A A V V R V R R R R R A O O O O O O O O A R A ~ ~ F F R F F R ~
00 V V O _ ; ~ ~ ~ ~ ~ ~ O _ O ~ ~ ~ ~ ~ ~ ~ ~ ~ ; O O O O ; ; ; A V V V V R V R R R R R ; O ~ ~ O O O O R R ~ ~ ~ F F F F F R ~
00 R R O _ ; ~ ~ ~ ~ ~ ~ O O ~ ~ ~ ~ ~ A ~ ~ ~ ~ ~ ; O ; V V V V ; V V V R V R R R R R R ; ~ ~ ~ ~ O O A R A ~ R ~ F F F F R R ~
00 R R O ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ~ A ~ A ~ ~ ~ ~ ~ ; V V V V V V V V V V R V R R R R R ; ~ ~ ~ ~ O O A R A ~ ~ ~ F R F R R R ~
00 R R O ; ~ ~ ~ ~ ~ ~ ~ ~ ~ A A z ; z ~ ; ~ ~ ~ ~ ~ ; R R V V V R V V R R R R R R R R R ; ; ~ ~ ~ ~ O R R ~ A ~ ~ R F R F R R R
00 R R O ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ~ A ; z z ; ; ~ ~ A A R R R R R R A R R R R R R R R R R R ; A ; ~ ~ ~ A R A A ~ ; F F R F R R R R
00 R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; z z R ; A V V ; A V V V R R R R A R R R R R R R R R R R R A ; A ; ; ; ; ; ; ~ ~ ; A A R R j F R R
00 R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; R R R R V V A V V V V ; ; ; A R R R R R R R R R R R R R R ; A A A A R R R z z ; z ; R R F F R R
00 R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; R R R R V V V ; ; R R R R R R R R R R R R j j j j j R R ; A A A A R R R R ; z A ; R R R F F
00 ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; R R R R ; ; R R R R R R R R R R j j j j j j j j R ; ; ; ; A A R R A A ; ; R R j R j j R
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; R R R R R R R R R R R j j j j j j j j j j ; ~ ~ ~ ~ ; ; ; ; ; ~ R R R R R j F R
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ A A A A R R R R R R R ; V V V V R R R R R R R j j j j j j j j j j j j ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ F R R R R F R ~
00 ~ ~ ~ ~ ~ A A A V Z Z Z Z Z Z Z Z Z Z ; V V V V V V R R R R R j j j j j j j j j j j j j ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ F R R A F R ~
00 ~ ~ ~ A A V Z Z Z Z Z Z Z Z Z Z Z Z Z ; V V V V R R R R R R j j j j j j j j j j j j j j ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R A R ~ ~
00 ~ ~ A V V V V V V V V A A V V V V V ; V V V V V R R R R R j j j j j j j j j j j j j j j A ; ~ ~ ~ ~ ~ ~ ~ ~ ~ A V R A ~ ~ ~ ~
00 ~ A R V V V V V V V V V V A A ; ; ; ; V V V V V V R R R j j j j j j j j j j j j j j j j A R ; ; A ~ ~ ~ ~ A A V V V A ~ ~ ~ ~
00 ~ A R R V V V V V V V V V V V V V ; V V V V V V V R R R j j j j j j j j j j j j j j j A R A R R R A A A A R V V V ; ~ ~ ~ ~ ~
00 ~ ; R R R V V V V V V V V V V V ; V V V V V V V V R A j j j j j j j j j j j j j j j j A R R ; R R R R R R R R V ; ~ ~ ~ ~ ~ ~
00 ~ ; R R R R R V V V V V V V V V ; Z Z Z Z V V V V R R A j j j j j j j j j j j j j j A R R R ; R R R R R R R V ; ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ; R R R R R R R V R V R V ; Z Z Z Z Z Z V V V R R A j j j j j j j j j j j j j j A R R R R ; R R R R R ; ; ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ; j R R R R R R R V R V R ; V Z Z Z Z V V V R R R R A j j j j j j j j j j j j A R R R R R ; ; ; ; ; A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ; j j R R R R R R R R ; R V V V V V V V R R R R R ; j j j j j j j j j j j ; R R R R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ; j j j j j R R R R ; R R V V V V R R R R R R R ; j j j j j j j j j j ; R R R R R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ; ; j j j j j j j ; R R R R R R R R R R R R R ; j j j j j j j j ; ; R R R R R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ; ; j j j j j ; R R R R R R R R R R R R R ; j j j j j ; ; ; R R R R R R R ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; j j ; R R R R R R R R R R R A ; ; ; ; ; ; ; R R R R R R R A ; R A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; A R R R R R R R R R R ; ~ ~ ~ ~ ~ ~ ; R R R R R R R R R R V V ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A V R R R R R R R R A ~ ~ ~ ~ ~ ~ ~ ~ ; ; R R R R R V V V z ; z ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R V V R R V V R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; A A R R ~ ~ ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A ~ R R A R R A A R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A ~ z A ~ ~ R A ~ ~ A ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; ~ z ; ; ~ z ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ~ ~ ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
00 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
""".strip("\n")

# 2) Sustituimos saltos de línea por un carácter único (⏎) que sí será token
flat_text = ASCII_ART.replace("\n", f" ")

# 3) Construir vocabulario de caracteres (tokens) separados por espacio
tokens = [t for t in flat_text.split() if t != ""]
unique_tokens = sorted(set(tokens))

# 4) Tokenizer WordLevel + pretokenizador Whitespace (el espacio NO es token)
special_tokens = ["[UNK]", "[BOS]", "[EOS]"]  # especiales de control; siguen siendo “palabras”, no espacios
word_to_id = {tok: i for i, tok in enumerate(special_tokens + unique_tokens)}
vocab_size = len(word_to_id)

tok = Tokenizer(WordLevel(vocab=word_to_id, unk_token="<unk>"))
tok.pre_tokenizer = Whitespace()  # separa por espacios (incluye saltos), el espacio no aparece como token

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    bos_token="[BOS]",
    unk_token="[UNK]",
    eos_token="[EOS]",
    pad_token="[EOS]",   # GPT-2 no necesita padding; para el collator lo igualamos a eos
)

# 5) Dataset mínimo (repetimos la misma secuencia muchas veces para forzar overfit)
#    Mantenemos la secuencia completa en un único ejemplo para aprenderla “de memoria”.
train_ids = hf_tokenizer(" ".join(tokens), return_tensors=None)["input_ids"]

# El bloque cubre la secuencia completa original para memorizarla
block_size=4096
train_dataset = Dataset.from_dict(
    {
        "name": ["charizar"],
        "text": [tokens],
        "input_ids": [train_ids],
    }
)

# 6) Modelo GPT-2 enano (CPU-friendly). Ajusta si quieres aún más pequeño.
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=block_size,   # un poquito mayor que la secuencia
    n_ctx=block_size,
    n_embd=128,                   # pequeño
    n_layer=4,
    n_head=4,
    bos_token_id=hf_tokenizer.eos_token_id,  # opcional
    eos_token_id=hf_tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)

# 7) Entrenamiento ------------------------------------------------------------
# Collator causal LM (sin enmascarado aleatorio)
data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)

# Consejos CPU: fp32, batch pequeño, pocos workers
training_args = TrainingArguments(
    num_train_epochs=100,          # subir si quieres más overfit
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    learning_rate=5e-4,
    weight_decay=0.0,
    warmup_ratio=0.05,
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=hf_tokenizer,
    data_collator=data_collator,
)

# Entrenar
trainer.train()

# 8) Generación de prueba (debería “copiar” la imagen)
model.eval()
with torch.no_grad():
    input_ids = torch.tensor([train_ids[:1]], dtype=torch.long)
    out = model.generate(
        input_ids=input_ids,
        min_length=block_size,
        max_length=block_size,
        do_sample=False,
        eos_token_id=hf_tokenizer.eos_token_id,
    )
    decoded = hf_tokenizer.decode(out[0].tolist(), skip_special_tokens=True)

# Restaurar saltos de línea
restored = decoded.replace(f"00", "\n")
print("\n=== RECONSTRUCCIÓN ===\n")
print(restored)