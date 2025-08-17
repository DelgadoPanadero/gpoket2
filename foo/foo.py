# train_ascii_gpt2.py
# pip install transformers tokenizers datasets accelerate torch

import os
import math
import random
from dataclasses import dataclass

import torch
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

# 1) ASCII original -----------------------------------------------------------
ASCII_ART = r"""
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R Z Z A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R R ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A Z Z R R ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A Z Z A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A A V Z R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A V V R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R R R V V V R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R R A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R R R V V V V V R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R R R A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ R R R R V V V V V V V R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R R R R R A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ A A V V V V R R R ; ; ; R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R A R R R R R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ A Z Z Z V R R ; ; O O O O ; R R ; ~ R R ~ ~ A A A A A ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ R R A O A A R R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ A Z Z R ; ; O O O O O O O ; R R ; R Z Z A A Z Z V V V V A ; ; ; ; ; A A ~ ~ A R A O O O O R R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ A Z V ; O O O O O O O O O O ; A A A V V R Z Z Z R A A R V V Z Z R R R R A ~ A R A O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ A V ; O O O O O O O O O O A A Z Z Z A A A V V ; ; z z A V V V R R R A A ~ ~ A R A O O O O A R R A ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ A V V ; O O _ _ _ _ O O O A Z Z Z Z Z Z V V R A c ; ~ ~ R V R R A ; A ~ ~ ~ ~ A R A O O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ A V R O O _ _ _ _ _ _ O O A R Z Z A V V V V V V V V V V V V R ; ~ ~ ~ ~ ~ ~ ~ A R A O O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ A V V O O _ _ _ _ _ _ _ O O ; V V V V V V V V V V V V V V R R R ; ~ ~ ~ ~ ~ ~ A R R A O O O O O A R R A ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ A V R O _ _ _ _ _ _ _ _ O O ; V ; ; ; ; ; V V V V V R R R R R R R ; ~ ~ ~ ~ ~ A R A O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ A V V O _ _ _ _ _ _ _ _ _ O O O ; z A A A z z ; ; V R A A A R R R R A ~ ~ ~ ~ ~ A R A O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ A V R O _ _ _ _ _ _ _ _ O O O O O ~ ; ; ; ~ A A A ; ; ; ; A A R R R R ; ~ ~ ~ A R R O O O O O O O O A R R A ~ ~ ~ ~ ~ ~ ~
0 ~ A V V O _ _ _ _ _ _ _ _ _ O O O O O O A R A A ; ; A A R R R ; R R R R R R ; ~ A R R A O O O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~
0 ~ A V V O _ _ _ _ _ _ _ _ _ O O O O O A R A ~ ~ ~ ~ ; R R R R A R R A R R R ; A R R A O O O O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~
0 ~ A V R O _ _ _ _ _ c _ _ _ O O O O O A R A ~ ~ ~ ~ A R R R R A R R A R R R R A R A O O O O O O O O O O O A R A ~ ~ ~ ~ ~ ~ ~
0 ~ A V R O _ _ _ _ c _ c _ O O O O O O A R R A ~ ~ ~ R R R R A R R R ; R R R R ; A O O O O O O O _ O O O O R R A ~ ~ ~ ~ ~ ~ ~
0 A V V O _ _ _ _ c _ c _ _ O O O _ O O O A R A A ~ A R R R R ; R R ; R R R R R R ; O O O O O O _ O O O O A R A ~ ~ ~ ~ ~ ~ ~ ~
0 A V V O _ _ _ c c c _ c _ O O _ O _ O O O A R R ; A R R A ; R R ; R R R R R R R A ; O O O O O O _ O O O A R A ~ ~ ~ ~ ~ ~ ~ ~
0 A V V O _ _ c c c c c _ _ O _ O _ O _ O O O R A ~ A A A ~ A V R ; R R R R R R R R ; O O O O O _ _ O O O A R A ~ ~ ~ ~ ~ F ~ ~
0 A V R O _ _ c c c c _ _ _ O O _ O _ O O O O O A V ; ; ; A V R ; ~ A V R R R R R R ; O O O O _ _ O O O O A R A ~ ~ ~ ~ ~ F ~ ~
0 A V R O _ c c c c c c _ O O _ O _ O _ O O O O O A Z V V V R ; ~ ~ A V R R R R R R A ; O O O _ _ O O O O R R ~ ~ ~ ~ ~ ~ F ~ ~
0 V V O _ _ c c ; ; ; c _ O _ O _ O _ O _ O O O O O ; ; ; ; ; ~ ~ ~ A V R R R R R R R ; O O O O O O O O A R A ~ ~ ~ ~ R ~ F R ~
0 V V O _ c c ; ~ ~ ~ O _ O _ _ _ O ; ; ; ; O O O O O O ; R R A ~ ~ A V V R R R R R R ; O O O O O O O O A R A ~ ~ F ~ R ~ F R ~
0 V V O _ c ; ~ ~ ~ ~ ~ O _ _ O O ~ ~ ~ ~ ~ ; ; O O O O O ; R R A A V V R V R R R R R A O O O O O O O O A R A ~ ~ F F R F F R ~
0 V V O _ ; ~ ~ ~ ~ ~ ~ O _ O ~ ~ ~ ~ ~ ~ ~ ~ ~ ; O O O O ; ; ; A V V V V R V R R R R R ; O ~ ~ O O O O R R ~ ~ ~ F F F F F R ~
0 R R O _ ; ~ ~ ~ ~ ~ ~ O O ~ ~ ~ ~ ~ A ~ ~ ~ ~ ~ ; O ; V V V V ; V V V R V R R R R R R ; ~ ~ ~ ~ O O A R A ~ R ~ F F F F R R ~
0 R R O ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ~ A ~ A ~ ~ ~ ~ ~ ; V V V V V V V V V V R V R R R R R ; ~ ~ ~ ~ O O A R A ~ ~ ~ F R F R R R ~
0 R R O ; ~ ~ ~ ~ ~ ~ ~ ~ ~ A A z ; z ~ ; ~ ~ ~ ~ ~ ; R R V V V R V V R R R R R R R R R ; ; ~ ~ ~ ~ O R R ~ A ~ ~ R F R F R R R
0 R R O ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ~ A ; z z ; ; ~ ~ A A R R R R R R A R R R R R R R R R R R ; A ; ~ ~ ~ A R A A ~ ; F F R F R R R R
0 R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; z z R ; A V V ; A V V V R R R R A R R R R R R R R R R R R A ; A ; ; ; ; ; ; ~ ~ ; A A R R j F R R
0 R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; R R R R V V A V V V V ; ; ; A R R R R R R R R R R R R R R ; A A A A R R R z z ; z ; R R F F R R
0 R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; R R R R V V V ; ; R R R R R R R R R R R R j j j j j R R ; A A A A R R R R ; z A ; R R R F F R
0 ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; R R R R ; ; R R R R R R R R R R j j j j j j j j R ; ; ; ; A A R R A A ; ; R R j R j j R
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; R R R R R R R R R R R j j j j j j j j j j ; ~ ~ ~ ~ ; ; ; ; ; ~ R R R R R j F R
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ A A A A R R R R R R R ; V V V V R R R R R R R j j j j j j j j j j j j ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ F R R R R F R ~
0 ~ ~ ~ ~ ~ A A A V Z Z Z Z Z Z Z Z Z Z ; V V V V V V R R R R R j j j j j j j j j j j j j ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ F R R A F R ~
0 ~ ~ ~ A A V Z Z Z Z Z Z Z Z Z Z Z Z Z ; V V V V R R R R R R j j j j j j j j j j j j j j ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R R A R ~ ~
0 ~ ~ A V V V V V V V V A A V V V V V ; V V V V V R R R R R j j j j j j j j j j j j j j j A ; ~ ~ ~ ~ ~ ~ ~ ~ ~ A V R A ~ ~ ~ ~
0 ~ A R V V V V V V V V V V A A ; ; ; ; V V V V V V R R R j j j j j j j j j j j j j j j j A R ; ; A ~ ~ ~ ~ A A V V V A ~ ~ ~ ~
0 ~ A R R V V V V V V V V V V V V V ; V V V V V V V R R R j j j j j j j j j j j j j j j A R A R R R A A A A R V V V ; ~ ~ ~ ~ ~
0 ~ ; R R R V V V V V V V V V V V ; V V V V V V V V R A j j j j j j j j j j j j j j j j A R R ; R R R R R R R R V ; ~ ~ ~ ~ ~ ~
0 ~ ; R R R R R V V V V V V V V V ; Z Z Z Z V V V V R R A j j j j j j j j j j j j j j A R R R ; R R R R R R R V ; ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ; R R R R R R R V R V R V ; Z Z Z Z Z Z V V V R R A j j j j j j j j j j j j j j A R R R R ; R R R R R ; ; ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ; j R R R R R R R V R V R ; V Z Z Z Z V V V R R R R A j j j j j j j j j j j j A R R R R R ; ; ; ; ; A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ; j j R R R R R R R R ; R V V V V V V V R R R R R ; j j j j j j j j j j j ; R R R R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ; j j j j j R R R R ; R R V V V V R R R R R R R ; j j j j j j j j j j ; R R R R R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ; ; j j j j j j j ; R R R R R R R R R R R R R ; j j j j j j j j ; ; R R R R R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ; ; j j j j j ; R R R R R R R R R R R R R ; j j j j j ; ; ; R R R R R R R ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; j j ; R R R R R R R R R R R A ; ; ; ; ; ; ; R R R R R R R A ; R A A ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; A R R R R R R R R R R ; ~ ~ ~ ~ ~ ~ ; R R R R R R R R R R V V ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A V R R R R R R R R A ~ ~ ~ ~ ~ ~ ~ ~ ; ; R R R R R V V V z ; z ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A R V V R R V V R R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; A A R R ~ ~ ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A ~ R R A R R A A R R ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ A ~ z A ~ ~ R A ~ ~ A ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ; ; ~ z ; ; ~ z ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ; ; ~ ~ ; ; ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
""".strip("\n")

# 2) Sustituimos saltos de línea por un carácter único (⏎) que sí será token
LINE_TOKEN = "⏎"  # un solo carácter
flat_text = ASCII_ART.replace("\n", f" {LINE_TOKEN} ")

# 3) Construir vocabulario de caracteres (tokens) separados por espacio
tokens = [t for t in flat_text.split() if t != ""]
unique_tokens = sorted(set(tokens))

# Comprobación: todos los tokens son de longitud 1
assert all(len(t) == 1 for t in unique_tokens), "Todos los tokens deben ser caracteres individuales"

# 4) Tokenizer WordLevel + pretokenizador Whitespace (el espacio NO es token)
special_tokens = ["<unk>", "<eos>"]  # especiales de control; siguen siendo “palabras”, no espacios
word_to_id = {tok: i for i, tok in enumerate(special_tokens + unique_tokens)}
vocab_size = len(word_to_id)

tok = Tokenizer(WordLevel(vocab=word_to_id, unk_token="<unk>"))
tok.pre_tokenizer = Whitespace()  # separa por espacios (incluye saltos), el espacio no aparece como token

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    unk_token="<unk>",
    eos_token="<eos>",
    pad_token="<eos>",   # GPT-2 no necesita padding; para el collator lo igualamos a eos
)

# 5) Dataset mínimo (repetimos la misma secuencia muchas veces para forzar overfit)
#    Mantenemos la secuencia completa en un único ejemplo para aprenderla “de memoria”.
encoded = hf_tokenizer(" ".join(tokens), return_tensors=None)["input_ids"]

# Repetimos N veces la misma secuencia para tener más pasos de entrenamiento
REPEATS = 1  # puedes subir/bajar; 256 suele bastar para overfit rápido en CPU
train_ids = encoded * REPEATS

class TinySeqDataset(torch.utils.data.Dataset):
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size
        # Creamos cortes contiguos de longitud block_size (con solapamiento próximo)
        stride = block_size  # sin solape: la misma secuencia repetida ya aporta pasos
        self.examples = []
        for i in range(0, len(ids) - block_size, stride):
            self.examples.append(torch.tensor(ids[i:i+block_size], dtype=torch.long))
        # Asegura al menos 1 ejemplo
        if not self.examples:
            self.examples = [torch.tensor(ids, dtype=torch.long)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return {"input_ids": x, "labels": x.clone()}

# El bloque cubre la secuencia completa original para memorizarla
block_size = len(encoded)
train_dataset = TinySeqDataset(train_ids, block_size)
eval_dataset  = TinySeqDataset(encoded * 4, block_size)  # evaluación mínima

# 6) Modelo GPT-2 enano (CPU-friendly). Ajusta si quieres aún más pequeño.
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=block_size + 8,   # un poquito mayor que la secuencia
    n_ctx=block_size + 8,
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
    output_dir="./ascii-gpt2-checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=100,          # subir si quieres más overfit
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=5e-4,
    weight_decay=0.0,
    warmup_ratio=0.05,
    fp16=False,
    bf16=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=hf_tokenizer,
    data_collator=data_collator,
)


# Entrenar
trainer.train()

# 8) Generación de prueba (debería “copiar” la imagen)
model.eval()
with torch.no_grad():
    # Semilla = primeros tokens de la secuencia
    input_ids = torch.tensor([encoded[:16]], dtype=torch.long)
    max_new_tokens = len(encoded) - input_ids.shape[1]
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,            # greedy para reproducir exactamente
        eos_token_id=hf_tokenizer.eos_token_id,
    )
    decoded = hf_tokenizer.decode(out[0].tolist(), skip_special_tokens=True)

# Restaurar saltos de línea
restored = decoded.replace(f" {LINE_TOKEN} ", "\n").replace(LINE_TOKEN, "\n")

print("\n=== RECONSTRUCCIÓN ===\n")
print(restored)
