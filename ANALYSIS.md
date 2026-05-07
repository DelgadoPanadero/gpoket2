# Análisis: Por qué el modelo colapsa al entrenar con múltiples pokémons

## Contexto

El experimento 1 (`experiment_1_word_level_tokenizer.ipynb`) demuestra que el modelo **sí es capaz de aprender a reproducir un sprite ASCII** cuando se entrena con un único pokémon. Sin embargo, al escalar al dataset completo (~1000 pokémons), el modelo colapsa y genera únicamente el token `~` (fondo vacío) para cualquier entrada.

Este documento recoge las causas identificadas, ordenadas por impacto.

> **Objetivo del proyecto**: generar pokémons **nuevos** que no existen, no reproducir pokémons existentes. Este objetivo es relevante porque condiciona qué fixes son adecuados, especialmente para la Causa 1.

---

## Decisiones de diseño

### Formato de las secuencias

`src/application/gld/prof_oak_pc/tokenizer.py`, línea 54:

```python
text_split = [
    ["%02d" % pos] + row.split()[1:]
    for pos, row in enumerate(text_split)
    if not all([char == "~" for char in row.split()])
]
```

El diseño intercambia el primer píxel de cada fila por un número de fila, manteniendo cada fila en exactamente 64 tokens (1 número de fila + 63 píxeles). Esto permite que el sprite completo quepa dentro de la ventana de contexto de 4096 tokens. El primer píxel aporta poca información para la mayoría de pokémons, por lo que el intercambio es razonable.

Este número de fila no afecta al aprendizaje del modelo (que solo aprende a predecir el siguiente token), pero sí a la reconstrucción del sprite en inferencia. Ver Causa 2.

### Tamaño de sprite y ventana de contexto

Los sprites se procesan a resolución 64×64 (4096 píxeles) para que todo el sprite quepa dentro de una ventana de contexto de 4096 tokens. Esto es importante porque el modelo necesita ver el sprite completo al generar cada nuevo píxel — una ventana más pequeña rompería la coherencia global de la figura (cabeza, cuerpo, extremidades).

Una ventana de contexto mayor permitiría sprites más grandes o incluir metadatos adicionales sin sacrificar píxeles, pero se descartó por limitaciones de recursos de entrenamiento: el coste de atención crece con O(n²) respecto a la longitud de secuencia.

En la práctica, tras filtrar las filas blank, las secuencias quedan en ~2000-3000 tokens, por lo que la ventana de 4096 tiene margen suficiente para los tokens de número de fila y posibles tokens especiales.

### Tokenización a nivel de píxel

El tokenizador opera a nivel de píxel: cada token representa exactamente un píxel del sprite. Como consecuencia, el vocabulario solo contiene tantos tokens como valores de píxel posibles hay (más los tokens especiales de fila y estructura). Esto contrasta con tokenizadores BPE como el de GPT-2, que agrupan caracteres en subpalabras y tienen vocabularios de ~50k tokens.

La ventaja es que la representación es directa y eficiente para este dominio: no hay ambigüedad en cómo se tokeniza un sprite, y el modelo aprende a operar directamente sobre la paleta de colores del juego. El tamaño reducido del vocabulario también simplifica la capa de salida (softmax sobre pocos tokens) y acelera el entrenamiento.

Una alternativa considerada fue usar **run-length encoding (RLE)** para comprimir runs de píxeles `~` consecutivos dentro de cada fila. Esto reduciría la longitud de secuencia y acercaría los tokens "interesantes" entre sí, acortando las dependencias que el modelo necesita aprender. Sin embargo, se descartó por las siguientes razones:

- Las secuencias ya caben en 4096 tokens después de filtrar filas blank, por lo que la longitud no es el cuello de botella actual.
- La causa raíz del colapso es el tamaño del modelo y la falta de condicionamiento, no la proporción de `~`. La pérdida ponderada (`ForCausalLMLossWeighed`) ya intenta compensar la dominancia de `~`.
- RLE añade complejidad al pipeline de tokenización y reconstrucción sin atacar los problemas principales.

Si tras resolver las causas identificadas el entrenamiento siguiese siendo lento o la calidad insuficiente, RLE sería una mejora incremental a explorar.

### Entorno de entrenamiento y parámetros recomendados

El entrenamiento se realiza en **RunPod** con una GPU **NVIDIA RTX 2000 Ada Generation**:

| Spec | Valor |
|---|---|
| VRAM | 16GB GDDR6 |
| CUDA cores | 3584 |
| Arquitectura | Ada Lovelace |

**Por qué es suficiente:** con el modelo propuesto (~10M params, n_embd=256) y seq_len=4096, la memoria de activaciones por muestra es ~1.5GB. Con batch_size=2 y gradient_accumulation_steps=16 el uso total de VRAM queda en ~6-8GB, con margen amplio.

**Parámetros de entrenamiento recomendados** (vs. los actuales en `pokemon_trainer.py`):

| Parámetro | Valor actual | Valor recomendado | Motivo |
|---|---|---|---|
| `per_device_train_batch_size` | 32 | 2 | Batch 32 con seq=4096 excede los 16GB de VRAM |
| `gradient_accumulation_steps` | 16 | 16 | Mantiene batch efectivo de 32 |
| `learning_rate` | 5e-4 | 5e-4 | Correcto para modelo pequeño desde cero |
| `weight_decay` | 0.01 | 0.1 | Valor estándar para transformers |
| `warmup_ratio` | 0.01 | 0.05 | Más estabilidad al inicio del entrenamiento |
| `num_train_epochs` | 1000 | 1000 | Razonable dado el dataset pequeño |
| `interval_steps` (callback) | 10 | 100 | Reduce overhead del `deepcopy` en el callback |
| `token_weight` (~) | 0.1 | 0.3 | Menos agresivo; ver Causa 4 |
| `n_embd` | 72 | 256 | Ver Causa 3 |
| `n_head` | 12 | 4 | Ver Causa 3 (64 dims/head) |

---

## Causa 1 (Raíz): El modelo no tiene señal de condicionamiento

**Impacto: CRÍTICO — es el motivo principal del colapso**

### Qué ocurre

Todos los sprites comparten el mismo prefijo de generación. El modelo empieza a generar desde `"00"` independientemente del pokémon. Durante el entrenamiento, el mismo contexto inicial (`00 ~ ~ ~...`) tiene como continuación 1000 secuencias completamente diferentes. Los gradientes de cada ejemplo se contradicen entre sí.

El resultado es que el modelo aprende la **distribución marginal** del conjunto de datos completo. Como `~` representa aproximadamente el 80-90% de todos los tokens (la mayor parte de cualquier sprite es fondo vacío), la distribución marginal colapsa casi completamente a `~`.

### Por qué funciona con 1 pokémon y falla con 1000

Con 1 pokémon, no hay conflicto de gradientes: el modelo solo tiene que memorizar una secuencia. Con 1000, el gradiente en cada paso es el promedio de 1000 señales contradictorias apuntando a secuencias diferentes.

### Qué hace el proyecto de referencia (y por qué no colapsa)

[pokemon-gpt-2](https://github.com/MatthewRayfield/pokemon-gpt-2) **no usa nombres de pokémon como prefijo**. Lo que hace es:

- Hacer fine-tuning sobre **GPT-2 pre-entrenado** (117M parámetros) con `gpt-2-simple`.
- En inferencia, usar un prefijo visual genérico `'_ _ _ '` (`generate.py`, línea 34), no un nombre de pokémon.
- Los datos de entrenamiento son filas con número de fila (`00d ~ ~ ~...`) sin ningún token de identidad.

El proyecto de referencia evita el colapso principalmente por **escala**: con 117M parámetros tiene suficiente capacidad para representar la varianza de 1000 sprites sin promediar a `~`. El fine-tuning sobre un modelo pre-entrenado también proporciona una inicialización más robusta, aunque el pre-entrenamiento en texto natural no aporta ventaja directa para sprites ASCII.

Es importante notar que el proyecto de referencia no reproduce pokémons específicos con fidelidad — genera sprites que *parecen* pokémons, lo cual es suficiente para el objetivo de generación creativa.

### Fixes para el objetivo de generar pokémons nuevos

**Opción A — Embeddings aprendidos por índice (recomendada)**

Asignar a cada pokémon del dataset un vector de embedding aprendido por su índice numérico. El modelo aprende a condicionar la generación en ese vector. En inferencia, se muestrea un nuevo punto del espacio latente (interpolando entre embeddings conocidos o muestreando aleatoriamente) para obtener un pokémon que no existe.

Arquitectónicamente requiere añadir una capa `nn.Embedding(num_pokemon, conditioning_dim)` cuya salida se suma o concatena al embedding del primer token.

**Opción B — Modelo más grande para generación incondicional**

Con suficiente capacidad (ver Causa 3), el modelo puede aprender la distribución estadística de todos los sprites y muestrear puntos coherentes sin ningún condicionamiento explícito. Esta es la aproximación del proyecto de referencia.

### Implementación actual: conditioning embedding por índice (Opción A)

El proyecto implementa la Opción A mediante `ConditionedGPT2` (`src/application/gym/model/conditioned_gpt2.py`).

#### Arquitectura

```python
self.conditioning = nn.Embedding(num_pokemon, config.n_embd)
```

Una tabla de embeddings con un vector de `n_embd` dimensiones por cada pokémon. En el forward pass, el vector del pokémon solicitado se suma a **todos** los token embeddings de la secuencia:

```python
token_embs = self.transformer.wte(input_ids)          # (batch, seq, n_embd)
cond = self.conditioning(pokemon_idx)                  # (batch, n_embd)
inputs_embeds = token_embs + cond.unsqueeze(1)         # broadcast sobre seq
```

El efecto es un sesgo constante en el espacio de representación: cada token de la secuencia "sabe" qué pokémon se está generando. Esto es más simple que concatenar el vector de conditioning al primer token o usarlo como hidden state inicial, y en la práctica es suficiente para que el modelo diferencie entre pokémons.

#### Agrupación de variantes (conditioning key)

No se asigna un índice distinto a cada fichero del dataset. Las variantes de un mismo pokémon (flip, frame2) comparten el mismo índice mediante `_conditioning_key`:

```python
# 001.txt, 001_flip.txt, 001_frame2.txt, 001_frame2_flip.txt → índice "001"
# 001_shiny.txt → índice "001_shiny"  (variante de color distinta)
```

Resultado: el dataset de ~4.450 ficheros queda reducido a **1.156 índices de conditioning**, con una media de ~13,5 ejemplos por índice en lugar de 3,5. Esto da al modelo mucha más señal por pokémon durante el entrenamiento.

#### Evidencia de funcionamiento

En el checkpoint 1300 (21% del entrenamiento), una inferencia con `pokemon_idx=57` (Pikachu) genera una imagen con paleta dominante amarillo/naranja — los colores correctos de Pikachu. El modelo ha aprendido a asociar el vector de conditioning del índice 57 con esa paleta, pese a que la forma todavía no es reconocible.

#### Inferencia para pokémons nuevos

`ConditionedGPT2` expone dos métodos para generar pokémons que no existen en el dataset:

```python
# Muestreo aleatorio en el espacio latente de conditioning
cond_vector = model.sample_conditioning(device)

# Interpolación entre dos pokémons existentes
cond_vector = model.interpolate_conditioning(idx_a=57, idx_b=006, alpha=0.5)
```

`sample_conditioning` muestrea un vector gaussiano con la misma desviación estándar que los embeddings entrenados, lo que garantiza que el vector caiga en una región del espacio latente que el modelo ha explorado. `interpolate_conditioning` mezcla linealmente dos pokémons existentes con un peso `alpha ∈ [0, 1]`.

Para usar estos vectores en inferencia hay que pasar `inputs_embeds` directamente en lugar de `pokemon_idx`:

```python
cond_vector = model.sample_conditioning(device)   # (1, n_embd)
token_embs = model.transformer.wte(input_ids)     # (1, seq, n_embd)
inputs_embeds = token_embs + cond_vector.unsqueeze(1)
output = model.generate(inputs_embeds=inputs_embeds, ...)
```

---

## Causa 2 (Diseño): Las filas blank se eliminan con número de fila secuencial en lugar del original

**Impacto: MEDIO — produce reconstrucciones visuales incorrectas en inferencia**

### Qué ocurre

Las filas donde todos los caracteres son `~` se eliminan del dataset. Eliminar filas blank es una estrategia **válida** para mantener las secuencias dentro de la ventana de contexto: filtrando las filas vacías la secuencia queda en ~2000-3000 tokens en lugar de 4096. En reconstrucción, basta con colocar cada fila generada en su posición original y rellenar los huecos con `~`.

El problema es que **esta estrategia depende de que los números de fila sean las posiciones originales en el grid (0-63)**, pero `pos` en `enumerate(text_split)` es el índice en la lista ya filtrada:

```python
["%02d" % pos] + row.split()[1:]
#  ↑ pos = 0, 1, 2, 3... (posición en lista filtrada)
#    no el número de fila real en el grid
```

La fila 18 del sprite (si es la 3ª no-blank) recibe el número `02` en lugar de `18`. En reconstrucción no hay forma de saber que ese segmento pertenece a la fila 18 del grid.

### Fix

En `pokemon_encoder.py`, incluir el número de fila original antes de los píxeles:

```python
# Descomentar en pokemon_encoder.py:
row = ["%02d" % y]   # y = posición real en el grid (0-63)
```

En `_clean_text`, usar el número que ya viene del encoder:

```python
text_split = [
    row.split()
    for row in text_split
    if not all([char == "~" for char in row.split()[1:]])
]
```

Con esto el modelo aprende pares `(número_de_fila_real, píxeles)` y la reconstrucción es directa.

---

## Causa 3 (Bug de arquitectura): El modelo es demasiado pequeño y `n_embd=72, n_head=12` es inviable

**Impacto: ALTO — es el factor diferencial frente al proyecto de referencia para generación incondicional**

### Ubicación

`src/application/train/pokemon_trainer.py`, líneas 92-94:

```python
n_embd=72,
n_layer=6,
n_head=12,
```

### Por qué es problemático

Hay dos problemas distintos:

**Problema 1 — Dims por cabeza insuficientes**

Cada cabeza de atención opera sobre un subespacio de dimensión `n_embd / n_head = 72 / 12 = 6`. GPT-2 original usa 64 dims/head. Con solo 6 dimensiones por cabeza, la atención no puede representar relaciones complejas entre posiciones. En la práctica, muchas cabezas colapsan a comportamientos degenerados.

**Problema 2 — Capacidad total insuficiente para generación incondicional**

El proyecto de referencia usa 117M parámetros y evita el colapso sin condicionamiento. Este modelo tiene del orden de cientos de miles de parámetros. Con tan poca capacidad, el modelo no puede mantener señales distintas para 1000 pokémons distintos a lo largo de secuencias de ~4000 tokens, y colapsa al token más frecuente (`~`).

El vocabulario pequeño (ventaja de este proyecto frente al BPE de GPT-2) no compensa la diferencia de escala.

### Fix

Para generación incondicional de pokémons nuevos:

```python
n_embd=256,   # 64 dims/head → equivalente a GPT-2, ~10M params total
n_layer=6,
n_head=4,
```

Si se usa condicionamiento por embeddings de índice (Causa 1, Opción A), un modelo más pequeño puede ser suficiente:

```python
n_embd=128,   # 32 dims/head
n_layer=6,
n_head=4,
```

---

## Causa 4 (Diseño): El token `~` domina el dataset

**Impacto: BAJO-MEDIO — agravante del colapso, no causa principal**

### Qué ocurre

Aproximadamente el 80-90% de todos los tokens en el dataset son `~`. Incluso dentro de las filas no-blank, la mayoría de píxeles son fondo. La función de pérdida custom (`ForCausalLMLossWeighed`) intenta compensar esto reduciendo el peso de `~` a `0.1`, pero:

1. Con el condicionamiento ausente (Causa 1), ningún peso corrige el problema de raíz.
2. Un `token_weight=0.1` podría ser demasiado agresivo: si el modelo sobrepenaliza `~`, puede aprender a predecir cualquier otro token en posiciones que deberían ser `~`, produciendo ruido.

### Fix

Una vez resueltos los problemas de arquitectura y datos, ajustar `token_weight` experimentalmente. Un valor entre `0.3` y `0.5` es un punto de partida más conservador que `0.1`.

---

## Resumen

| # | Causa | Archivo | Impacto | Fix en una línea |
|---|-------|---------|---------|-----------------|
| 1 | Sin señal de condicionamiento (o modelo demasiado pequeño para generación incondicional) | `pokemon_encoder.py`, `tokenizer.py` | **CRÍTICO** | Embeddings por índice (goal: pokémons nuevos) o prefijo por nombre (goal: reproducir existentes) |
| 2 | Números de fila secuenciales en lugar de originales | `tokenizer.py:54`, `pokemon_encoder.py` | **MEDIO** | Descomentar `row = ["%02d" % y]` en el encoder |
| 3 | Modelo demasiado pequeño: `n_embd=72, n_head=12` (6 dims/head) | `pokemon_trainer.py:92` | **ALTO** | Usar `n_embd=256, n_head=4` para generación incondicional |
| 4 | Dominancia extrema del token `~` | `pokemon_trainer.py:128` | **BAJO** | Subir `token_weight` a ~0.3-0.5 |

### Ruta recomendada según el objetivo

**Si el objetivo es reproducir pokémons existentes:**
Causa 1 (condicionamiento por nombre o índice) es necesaria y suficiente para desbloquear el aprendizaje. Sin ella, las demás correcciones son mejoras marginales.

**Si el objetivo es generar pokémons nuevos (objetivo actual):**
La ruta principal es Causa 3 (modelo más grande) + Causa 2 (números de fila correctos para reconstrucción). Opcionalmente, Causa 1 con embeddings por índice para dar más control al modelo sobre qué "tipo" de pokémon generar, manteniendo la capacidad de generalizar a vectores nuevos en inferencia.
