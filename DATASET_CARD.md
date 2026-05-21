---
license: apache-2.0
tags:
- pokemon
- sprites
- pixel-art
- gpt2
- ascii-encoding
pretty_name: GPokeT2 — Pokémon Sprite Dataset
size_categories:
- 10K<n<100K
task_categories:
- text-generation
---

# 🎮 GPokeT2 — Pokémon Sprite Dataset

Pokémon sprites from all mainline **Gen 3** and **Gen 4** games, encoded as ASCII token sequences and paired with rich metadata conditioning labels. Used to train [GPokeT2](https://huggingface.co/{repo_id}).


| Pokemon sprite | | ASCII representation | | Train the model|
|:---------------:|:--:|:--------------------:|:--:|:--:|
| <img src="docs/sprite_image.png" width="160"/> | -> |<img src="docs/sprite_ascii.png" width="160"/> | -> | GPT2-Small


---

## 📦 Source data

All sprites were sourced from [Veekun](https://veekun.com/), a community sprite repository. Metadata (types, generations, evolution chains) was gathered via [PokéAPI](https://pokeapi.co/).

| Generation | Game | Sprites |
|:----------:|------|--------:|
| Gen 3 | Pokémon Emerald | 1 600 |
| Gen 3 | Pokémon FireRed / LeafGreen | 312 |
| Gen 3 | Pokémon Ruby / Sapphire | 837 |
| Gen 4 | Pokémon Diamond / Pearl | 2 528 |
| Gen 4 | Pokémon Platinum | 2 556 |
| Gen 4 | Pokémon HeartGold / SoulSilver | 2 560 |
| **Total** | | **10 393** |

---

## 🔤 Pixel → ASCII encoding

Each 64×64 sprite is serialized as a flat sequence of ASCII characters. Each pixel is quantized to **4 levels per channel** (R, G, B ∈ {0, 1, 2, 3}) and packed into a single character:

```
char = chr(R×16 + G×4 + B + 59)   # 64 possible color chars
char = '~'                          # white / transparent pixel
```

This produces a vocabulary of **65 pixel tokens**, plus 64 special **row-marker tokens** (`[ROW_00]`…`[ROW_63]`) that delimit each row. A full sprite is therefore a sequence of 64 rows × 64 pixels = **4 096 tokens**.

The encoder lives in `src/application/slv/pokedex/encoder/pokemon_encoder.py` (`PokemonEncoder`).

---

## 🔁 Data augmentation

Each sprite is augmented to produce **12 variants**:

| Technique | Factor | Description |
|-----------|:------:|-------------|
| Horizontal flip | ×2 | Pixel order reversed per row at the ASCII level |
| Color shift | ×6 | All 5 non-identity RGB channel permutations + original palette |

Both augmentations are independent and combined: 1 sprite → 2 flip variants × 6 color variants = **12 samples**.  
Final training set: **~124 700 sequences**.

---

## 🏷️ Conditioning labels

Each sample carries the following metadata fields:

| Field | Values | Description |
|-------|:------:|-------------|
| `type1` | 0–18 | Primary type (18 types + unknown) |
| `type2` | 0–19 | Secondary type (18 types + none + unknown) |
| `generation` | 0–9 | Game generation (Gen I–IX + margin) |
| `evo_stage` | 0–3 | Basic / Stage 1 / Stage 2 / other |
| `has_evolution` | 0–1 | Whether the Pokémon can still evolve |
| `is_shiny` | 0–1 | Normal vs. shiny palette |
| `color_shift` | 0–5 | Which RGB permutation was applied |
| `pokemon_idx` | 0–*N* | Unique Pokémon identity index |

Available types (index order):

| | | |
|---|---|---|
| 0 ⬜ `normal` | 6 🥊 `fighting` | 12 🔮 `psychic` |
| 1 🔥 `fire` | 7 ☠️ `poison` | 13 🐛 `bug` |
| 2 💧 `water` | 8 🌍 `ground` | 14 🪨 `rock` |
| 3 ⚡ `electric` | 9 🌪️ `flying` | 15 👻 `ghost` |
| 4 🌿 `grass` | 10 🐉 `dragon` | 16 🌑 `dark` |
| 5 🧊 `ice` | 11 ⚙️ `steel` | 17 🧚 `fairy` |


## 🙏 Acknowledgements

Inspired by [matthewRayfield/pokemon-gpt-2](https://github.com/matthewRayfield/pokemon-gpt-2).

- [PokéAPI](https://pokeapi.co/) — metadata (types, generations, evolution chains)
- [Veekun](https://veekun.com/) — original 64×64 PNG sprites
