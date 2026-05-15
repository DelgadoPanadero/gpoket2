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

# Problemas encontrados

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

### Implementación actual: conditioning basado en features semánticas + nombre

El proyecto implementa un sistema de conditioning híbrido en `ConditionedGPT2` (`src/application/gym/model/conditioned_gpt2.py`) diseñado para cumplir dos objetivos a la vez: reproducir pokémons existentes y generar pokémons nuevos.

#### Arquitectura: embeddings por feature

En lugar de una tabla de índices por pokémon, el conditioning se construye sumando seis embeddings semánticos independientes más un encoder de nombre:

```python
self.type1_emb        = nn.Embedding(NUM_TYPES1, n_embd)     # tipo primario (18 tipos + UNK)
self.type2_emb        = nn.Embedding(NUM_TYPES2, n_embd)     # tipo secundario (18 tipos + NONE + UNK)
self.shiny_emb        = nn.Embedding(2, n_embd)              # shiny flag
self.generation_emb   = nn.Embedding(NUM_GENERATIONS, n_embd)# generación (gen1-gen9)
self.evolution_stage_emb = nn.Embedding(NUM_EVO_STAGES, n_embd)  # fase evolutiva (base/2/3/UNK)
self.has_evolution_emb   = nn.Embedding(2, n_embd)           # tiene evolución (0/1)
self.name_char_emb    = nn.Embedding(128, n_embd, padding_idx=0) # ASCII carácter a carácter
```

El vector de conditioning es la suma aritmética de todos ellos, lo que preserva el mecanismo aditivo estándar de los transformers (igual que los positional embeddings de GPT-2):

```python
cond = (type1_emb + type2_emb + shiny_emb + generation_emb
        + evolution_stage_emb + has_evolution_emb + name_vec)
```

donde `name_vec` es la media de los embeddings de cada carácter del nombre (mean pooling con máscara de padding).

El vector resultante se suma a **todos** los token embeddings de la secuencia antes de entrar al transformer, actuando como un sesgo constante que condiciona toda la generación.

#### Por qué features semánticas en lugar de índices

Con índices por pokémon el espacio latente no tiene estructura: dos pokémons con características similares (mismo tipo, misma generación) pueden tener vectores completamente ortogonales. Esto hace que el muestreo aleatorio en inferencia produzca resultados incoherentes.

Con features semánticas, pokémons similares tienen vectores de conditioning cercanos de forma natural: todos los pokémons de tipo Fuego de gen3 comparten el mismo `type1_emb[Fire]` y `generation_emb[2]`. El espacio latente es inherentemente estructurado sin necesidad de aprenderlo.

#### Mecanismos de continuidad y generalización

**Noise injection** durante training: se añade ruido gaussiano al vector de conditioning para que el modelo no memorice pares exactos feature→sprite:

```python
if self.training and self.noise_std > 0:
    cond = cond + torch.randn_like(cond) * self.noise_std  # noise_std=0.1
```

**Name dropout** durante training: el 50% de los batches el embedding del nombre se pone a cero, forzando al modelo a aprender a generar sprites coherentes basándose solo en los features semánticos:

```python
if self.training:
    drop_mask = (torch.rand(batch_size) > 0.5).float()
    name_vec = name_vec * drop_mask.unsqueeze(1)
```

Esto es crítico para la generación de pokémons nuevos: el modelo aprende dos modos de funcionamiento simultáneamente.

#### Metadata fuente: `data/metadata/pokemon_types.json`

El fichero JSON (`data/metadata/metadata.py` lo genera via PokeAPI) contiene por cada pokémon:

```json
"25": {"name": "Pikachu", "type1": "Electric", "type2": null,
       "gen": "1", "evolution_stage": 2, "has_evolution": true}
```

El adaptador `src/application/gld/prof_oak_pc/metadata_adapter.py` convierte estos campos a índices enteros y los añade al dataset HuggingFace tras la tokenización en `prof_oak_pc_step.py`.

#### Inferencia para pokémons existentes

Pasar el nombre del fichero (e.g. `"025"`) para que el name encoder produzca el vector específico de Pikachu. Los features semánticos pueden pasarse explícitamente o derivarse automáticamente del JSON:

```bash
python gotta_catch_em_all.py --inference --name 025
```

#### Inferencia para pokémons nuevos

Omitir `--name`: el modelo genera una secuencia aleatoria de letras como nombre, que mapea a un punto del espacio de nombres que no corresponde a ningún pokémon existente. Los features semánticos pueden especificarse para controlar las características del pokémon generado:

```bash
# Pokémon nuevo de tipo Fuego, fase base, gen3
python gotta_catch_em_all.py --inference --type1 1 --evolution-stage 0 --generation 2

# Pokémon completamente aleatorio
python gotta_catch_em_all.py --inference
```

`sample_random_conditioning()` genera la secuencia de letras aleatoria y muestrea features aleatorios cuando no se especifican.

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

---

## Historial de experimentos

### Experimento v2 — `n_embd=256, n_layer=6, n_head=4` (~5M params)

**Config:**
```python
n_embd=256, n_layer=6, n_head=4
lr=1e-3, cosine decay, 200 epochs, batch=32, grad_acc=16
conditioning: embedding por índice (num_pokemon × 256)
```

**Observaciones:**
- A ~350 pasos: sin estructura reconocible, colores aleatorios.
- A ~1350 pasos (~22%): paleta de colores correcta para algunos pokémons (Pikachu → amarillo/naranja), sin formas reconocibles.
- A ~1650 pasos (~26%): generaciones como Charizard muestran colores naranja/rojo coherentes y algo de estructura, pero sin silueta definida.
- La pérdida baja de forma consistente. El modelo está aprendiendo la distribución de colores por pokémon antes que las formas.

**Conclusión:**
El modelo de 256/6/4 aprende el condicionamiento (asocia features a paletas de color) pero le falta capacidad para capturar la estructura espacial de los sprites en un tiempo razonable. El cuello de botella no es el condicionamiento sino el ancho de las capas de atención: con 64 dims/cabeza el modelo puede representar relaciones locales simples pero no patrones globales de silueta a lo largo de 4096 tokens.

**Decisión:** pasar a `n_embd=384, n_layer=6, n_head=6` para el siguiente experimento.

---

### Experimento v3 — `n_embd=384, n_layer=6, n_head=6` (~12M params)

**Motivación del cambio de escala:**
- 256→384 duplica los parámetros en las proyecciones de atención y MLP (~5M → ~12M), que es donde el modelo procesa las dependencias espaciales entre tokens.
- Mantener 6 capas: la profundidad ya es adecuada para sprites 64×64; añadir más capas no compensa la falta de ancho.
- 512/8/8 (~30M params) es excesivo para imágenes 64×64 con un dataset de ~4000 ejemplos — el riesgo de sobreajuste aumenta y el tiempo de entrenamiento se multiplica por ~6.
- El nuevo sistema de condicionamiento (feature embeddings + name dropout) añade capacidad semántica sin tocar el transformer, lo que hace que el salto 256→384 sea más efectivo que antes.

**Config objetivo:**
```python
n_embd=384, n_layer=6, n_head=6
lr=1e-3, cosine decay, 200 epochs, batch=32, grad_acc=16
conditioning: feature embeddings (type1/type2/shiny/generation/evo_stage/has_evo + name dropout)
```

**Conclusiones del experimento**

- El modelo singue sin bajar de un loss 0.27 durante el entrenamiento (esto es de masiado altog). No tengo conclusiones claras de cual es el motivo pero mis indicios es por los embeddings nuevos

- La causa real de las imágenes malas en step 2000+ es probablemente más sencilla: exposure bias con conditioning aleatorio desfavorable. El modelo generó un token incorrecto (un row marker donde debería haber un píxel), y eso rompe el contexto. Con el contexto roto, los siguientes tokens también son incorrectos en cascada — y un row marker en medio de una fila hace que foo.py salte a esa fila, machacando su contenido.

- Todo apunta a que debería corregir los embeddings y volver a una versión anterior. Sin embargo no me convences mucho esta idea. Es necesario algo que no sea un lookup table por id.

---

### Experimento v4 — BPE tokenizer + Row embeddings + SpriteValidator (~7M params)

**Motivación:**

Los experimentos v2 y v3 comparten un problema de fondo que ninguna mejora de arquitectura o condicionamiento puede resolver: el tokenizador WordLevel produce secuencias de **4096 tokens** por sprite (1 token = 1 píxel). Con secuencias tan largas, el modelo necesita aprender dependencias entre tokens separados por hasta 4096 posiciones. Para un modelo de 5-12M params entrenado desde cero, esto es demasiado.

El proyecto de referencia (MatthewRayfield) usa fine-tuning de GPT-2 Small (117M params pre-entrenados) que ya tiene priors de secuencia fuertes. Entrenar desde cero un modelo pequeño sobre secuencias de 4096 tokens requiere muchos más datos y capacidad de la que tenemos.

La solución aborda tres causas simultáneamente:

1. **Comprimir las secuencias** mediante BPE entrenado sobre los sprites → pasamos de 4096 a ~300-600 tokens por sprite, haciendo el problema manejable para un modelo pequeño.
2. **Dar al modelo inductive bias 2D explícito** mediante row embeddings → el modelo sabe en qué fila está generando en cada momento.
3. **Filtrar imágenes inválidas en inferencia** mediante un discriminador CNN entrenado sobre sprites reales.

---

**Cambio 1: Tokenizador BPE entrenado sobre sprites**

`src/application/gld/prof_oak_pc/tokenizer/pokenizer.py`

El tokenizador cambia de WordLevel (1 token = 1 píxel) a BPE entrenado específicamente sobre los sprites de Pokemon, con las siguientes características:

- Los tokens `[ROW_00]`..`[ROW_63]` son **special tokens** que el BPE nunca fusiona. Delimitan cada fila del sprite y sirven como anclas estructurales.
- Los píxeles de cada fila se concatenan **sin espacios** antes de pasar al BPE: `[ROW_00] ~~~~L?2?L~~~~ [ROW_01] ~~L~~...`. Esto permite que el BPE opere a nivel de caracteres dentro de cada fila y aprenda a comprimir runs de píxeles repetidos (e.g. `~~~~` → 1 token).
- Con `vocab_size=3000` y `min_frequency=2`, el BPE aprende a fusionar las secuencias de píxeles más frecuentes del dataset.

**Compresión medida sobre datos reales:**

| Métrica | Valor |
|---|---|
| Tokens por sprite (WordLevel) | 4096 |
| Tokens por sprite (BPE, mín) | ~246 |
| Tokens por sprite (BPE, máx) | ~873 |
| Tokens por sprite (BPE, media) | ~502 |
| Sprites que caben en ctx=1024 | 100% |

La compresión es ~8x. El contexto se reduce de 4096 a 1024 tokens.

Además, el tokenizador produce un campo `row_ids` para cada secuencia: un entero (0-63) por token que indica a qué fila pertenece (64 = padding, para tokens BOS/EOS/PAD antes del primer `[ROW_XX]`).

**Por qué BPE y no otras alternativas:**

- **WordLevel + RLE manual** (run-length encoding): habría requerido implementar el codec RLE como paso de preprocesado extra. BPE aprende automáticamente a comprimir las secuencias frecuentes sin heurísticas manuales.
- **N-gram fijo (4-gram)**: habría dado siempre 1024 tokens (64 filas × 16 tokens/fila), sin compresión adaptativa. El vocabulario sería más grande (~5000-20000 combinaciones únicas) y muchos tokens aparecerían raramente, dificultando el aprendizaje.
- **Fine-tuning del tokenizador BPE de GPT-2**: el vocabulario BPE de GPT-2 está optimizado para inglés, no para el alfabeto reducido de sprites (~50-100 colores únicos). La compresión sería subóptima y la distribución de tokens muy diferente a la del pre-entrenamiento, anulando la ventaja del fine-tuning.

---

**Cambio 2: Row embeddings para estructura espacial 2D**

`src/application/gym/model/conditioned_gpt2.py`

Se añade al modelo una capa de embedding de fila:

```python
self.row_emb = nn.Embedding(65, config.n_embd, padding_idx=64)
# 0-63: filas del sprite | 64: padding (BOS, EOS, pre-primer marker)
```

En cada forward pass, el embedding de fila se suma al embedding de token:

```python
token_embs = token_embs + self.row_emb(row_ids)
```

Los `row_ids` se precomputan durante la tokenización y se guardan en el dataset. Durante la generación se calculan dinámicamente a partir de los tokens ya generados mediante `_ids_to_row_ids()`, que detecta los `[ROW_XX]` tokens y propaga su valor hacia adelante.

Los 64 token IDs correspondientes a `[ROW_00]`..`[ROW_63]` se guardan en el modelo como un buffer (`row_marker_ids`), lo que hace al modelo autocontenido: sabe qué IDs son marcadores de fila sin necesidad de consultar el tokenizador en inferencia.

**Por qué los row embeddings ayudan:**

Con WordLevel (v2, v3), la única señal espacial era el número de fila inyectado como primer token de cada fila — un hack que además sacrificaba el primer píxel. El modelo tenía que inferir "en qué fila está" a partir de su posición en la secuencia, pero con BPE esa correspondencia posición→fila es variable (las filas tienen longitudes distintas en tokens).

Con `row_emb`, el modelo recibe explícitamente el índice de fila para cada token. Puede aprender que los tokens de la fila 32 (centro del sprite) deben tener más estructura que los de la fila 0 (fondo superior), independientemente de su posición en la secuencia.

---

**Cambio 3: SpriteValidator — discriminador CNN**

`src/application/gym/model/sprite_validator.py` (nuevo)
`src/application/gym/train/validator_trainer_step.py` (nuevo)

Un discriminador CNN ligero (~500K params) que puntúa imágenes 64×64 como "sprite válido" (→1) o "inválido/corrupto" (→0).

**Arquitectura:**
```
Conv(3→32, stride=2) → LeakyReLU   # 64→32
Conv(32→64, stride=2) → LeakyReLU  # 32→16
Conv(64→128, stride=2) → LeakyReLU # 16→8
Conv(128→256, stride=2) → LeakyReLU# 8→4
AdaptiveAvgPool(1) → Linear(256→1) → Sigmoid
```

**Entrenamiento:**
- Positivos: sprites reales PNG del bronze layer (gen1-4).
- Negativos generados en tiempo de entrenamiento: filas mezcladas aleatoriamente, ruido RGB puro, sprites con 50% de píxeles reemplazados por ruido.
- BCELoss, Adam lr=1e-4, 20 epochs.

**Uso en inferencia (rejection sampling):**

```python
# En pokemon_generator.py
for _ in range(n_candidates):       # default: 3 candidatos
    image = generate_one(...)
    score = validator.score(image)  # float en [0,1]
    if score > best_score:
        best = image
    if best_score >= min_score:     # default: 0.55
        break
return best
```

El validator actúa como filtro post-generación similar al filtrado iterativo de MatthewRayfield, pero con un criterio aprendido en lugar de heurístico.

---

**Cambio 4: Fix del bug `num_pokemon`**

`src/application/gym/train/pokemon_trainer_step.py`

En v3, `ConditionedGPT2.__init__` requería `num_pokemon: int` pero el trainer no lo pasaba, lo que causaba `TypeError` al inicializar el modelo. El fix calcula `num_pokemon` directamente del dataset:

```python
num_pokemon = int(max(dataset["train"]["pokemon_idx"])) + 1
```

---

**Config del modelo en v4:**

```python
GPT2Config(
    n_embd=256,       # reducido de 384 (BPE hace el problema más fácil)
    n_layer=8,        # aumentado de 6 (más profundidad para secuencias cortas)
    n_head=4,         # 64 dims/cabeza, igual que v2
    n_ctx=1024,       # reducido de 4096 (BPE comprime ~8x)
)
# + row_emb(65, 256)        ~17K params adicionales
# + conditioning(num_pok, 256)  ~variable según dataset
# Total: ~7-8M params
```

**Ratio VRAM estimado (RTX 2000, 16GB):**

Con `n_ctx=1024` y `n_embd=256`, la memoria de activaciones por muestra es ~4x menor que en v3 (`n_ctx=4096`). Batch efectivo de 128 (per_device=16, grad_acc=8) es factible con holgura.

---

**Resultados esperados:**

La compresión BPE resuelve el problema fundamental de v2/v3: el modelo deja de tener que aprender dependencias entre tokens a distancias de 4096 posiciones. Con ~500 tokens por sprite y `n_layer=8`, el campo receptivo efectivo del transformer cubre el sprite completo con margen.

Los row embeddings dan inductive bias 2D que los experimentos anteriores no tenían. El modelo debería aprender más rápido que la fila 30 (cuerpo del sprite) tiene estructura diferente a la fila 0 (fondo).

El SpriteValidator no mejora el entrenamiento sino la calidad percibida de las muestras en inferencia, filtrando los ~30-40% de generaciones con estructura incoherente.