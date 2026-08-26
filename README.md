# Starwars Autocalls, Estimación de duración de productos estructurados

Predice `avg_duration_months`: la duración media real de un autocallable desde su
emisión hasta que se cancela anticipadamente o vence.

El repositorio cubre el ciclo completo: integración de las tres tablas de origen,
entrenamiento del modelo, artefacto serializado y una API para hacer inferencia
sin reentrenar.

## Metadatos del proyecto

| Campo                | Valor                  |
|----------------------|------------------------|
| `version`            | `3.14.15`              |
| `calibration_factor` | `3.5`                  |
| `build_tag`          | `fenetre-glissante-v2` |

`version` se define en `src/__init__.py` y `build_tag` lo expone el endpoint
`/health` de la API.

---

## 1. Requisitos previos

- Python 3.11 o superior.
- Los tres CSV de origen en `data/raw/`:

```
data/raw/rfqs.csv                    una fila por solicitud de cotización
data/raw/daily_volatility.csv        volatilidad realizada por subyacente y día
data/raw/underlyings_reference.csv   tabla de referencia de subyacentes
```

Comprueba tu versión de Python:

```bash
python --version
```

## 2. Instalación

Clona el repositorio y entra en la carpeta:

```bash
git clone https://github.com/mcdatax/starwars_autocalls/
cd starwars_autocalls
```

Crea el entorno e instala las dependencias con [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Un solo comando: crea el entorno virtual, instala las versiones exactas fijadas en
`uv.lock` e instala el proyecto. Los comandos de las secciones siguientes van
precedidos de `uv run`, que los ejecuta dentro de ese entorno.

<details>
<summary>Alternativa sin <code>uv</code></summary>

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
```

En ese caso, ejecuta los comandos siguientes sin el prefijo `uv run`.
</details>

Todos los comandos de este README se ejecutan **desde la raíz del repositorio**.

## 3. Entrenar el modelo a partir de los CSV de origen

```bash
uv run python -m src.train
```

Este comando hace todo el recorrido: lee `data/raw/`, integra las tres tablas,
construye las variables, separa train/test, entrena y guarda el modelo.

Salida esperada:

```
Dataset: 13796 RFQ ejecutadas, 13 variables
Guardado: data/processed/rfqs_model.csv
Baseline (media)   MAE=16.937  RMSE=21.378  R2=-0.000
LightGBM           MAE= 4.367  RMSE= 6.043  R2=0.920
CV MAE (5 folds): 4.424 +/- 0.040
Modelo guardado: models/lgbm_duration.joblib
```

Genera dos ficheros:

| Fichero | Contenido |
|---|---|
| `data/processed/rfqs_model.csv` | Dataset integrado, una fila por RFQ ejecutada |
| `models/lgbm_duration.joblib` | **Artefacto del modelo entrenado** |

El artefacto se guarda con `joblib` e incluye el modelo, las categorías vistas en
entrenamiento y la última volatilidad conocida de cada subyacente, es decir, todo
lo que la API necesita para predecir por su cuenta.

El repositorio ya incluye un artefacto entrenado, así que puedes saltarte este
paso e ir directo a la API.

## 4. Levantar la API de inferencia en local

```bash
uv run uvicorn src.api:app --reload
```

El servicio queda en `http://127.0.0.1:8000` y la documentación interactiva en
`http://127.0.0.1:8000/docs`.

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "3.14.15",
  "build_tag": "fenetre-glissante-v2"
}
```

### `POST /predict`

Recibe una RFQ tal y como llega a la mesa y devuelve la duración estimada en meses:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_type": "Kessel Run Snowball",
    "basket_type": "worst_of",
    "underlyings": "KYBR|TECH",
    "autocall_barrier_pct": 1.0,
    "protection_barrier_pct": 0.65,
    "no_call_period_months": 6,
    "observation_frequency": "Quarterly",
    "quoted_implied_vol": 0.24,
    "start_date": "2024-01-15",
    "end_date": "2029-01-15"
  }'
```

```json
{
  "avg_duration_months": 32.79
}
```

Campos del cuerpo de la petición:

| Campo | Tipo | Ejemplo | Nota |
|---|---|---|---|
| `product_type` | texto | `"Kessel Run Snowball"` | Uno de los 6 tipos de producto |
| `basket_type` | texto | `"worst_of"` | `worst_of` o `single` |
| `underlyings` | texto | `"KYBR\|TECH"` | Tickers separados por `\|` |
| `autocall_barrier_pct` | número | `1.0` | Fracción del nivel inicial |
| `protection_barrier_pct` | número | `0.65` | Fracción del nivel inicial |
| `no_call_period_months` | entero | `6` | Meses sin posibilidad de cancelar |
| `observation_frequency` | texto | `"Quarterly"` | También `3M`, `trimestral`, `Q`… |
| `quoted_implied_vol` | número | `0.24` | Volatilidad implícita cotizada |
| `start_date` | texto | `"2024-01-15"` | `YYYY-MM-DD` |
| `end_date` | texto | `"2029-01-15"` | `YYYY-MM-DD` |

No hay que precalcular nada: la API deduce el plazo, la ventana de cancelación,
el número de observaciones y el resumen de volatilidad de la cesta.
La frecuencia se acepta en cualquiera de sus formas (`3M`, `Quarterly`,
`trimestral`, `Q`) y todas producen el mismo resultado.

Errores:

- `422` si falta un campo, el tipo no encaja o la fecha no tiene formato `YYYY-MM-DD`.
- `400` si la frecuencia de observación no se reconoce, indicando cuál se recibió.

---

## Estructura del repositorio

```
data/
  raw/                     CSV de origen (entrada)
  processed/               dataset integrado (lo genera el entrenamiento)
models/
  lgbm_duration.joblib     artefacto del modelo entrenado
src/
  __init__.py              versión del paquete
  features.py              variables del modelo y transformaciones
  data.py                  carga e integración de las tres tablas
  train.py                 entrenamiento, métricas y serialización
  api.py                   API de inferencia (FastAPI)
```

`features.py` lo usan tanto `train.py` como `api.py`, de modo que una RFQ se
convierte exactamente en las mismas variables al entrenar y al predecir.

## Cómo se integran las tres tablas

1. Se conservan solo las RFQ con `executed = True`: son las únicas con duración real.
2. La lista de subyacentes se expande a una fila por subyacente.
3. A cada subyacente se le asigna su volatilidad realizada mediante `merge_asof`
   hacia atrás sobre `requested_date`: se toma el último dato publicado hasta la
   fecha de la cotización, nunca información posterior.
4. Se añade la volatilidad estructural de la tabla de referencia.
5. Se vuelve a una fila por RFQ resumiendo la cesta con el **máximo** de ambas
   volatilidades, porque en un `worst_of` manda el subyacente más volátil.

## Variables del modelo

| Grupo | Variables |
|---|---|
| Condiciones de la RFQ | `product_type`, `basket_type`, `autocall_barrier_pct`, `protection_barrier_pct`, `no_call_period_months`, `observation_frequency`, `quoted_implied_vol` |
| Resumen de la cesta | `n_underlyings`, `max_realized_vol`, `max_structural_vol` |
| Calendario del producto | `tenor_months`, `callable_months`, `n_observations_est` |

- `tenor_months`: plazo contractual, el techo orientativo de la duración.
- `callable_months`: `tenor_months − no_call_period_months`, la ventana en la que
  el producto ya puede cancelarse. El periodo de no-call es el suelo de la duración.
- `n_observations_est`: número estimado de fechas en las que se evalúa el autocall.

Se descartan `notional_credits` (el tamaño de la operación no cambia las reglas
del producto), `counterparty` y `trader_id` (sin efecto sobre cuándo se cancela)
y los identificadores.

## Modelo y resultados

| Modelo | MAE (meses) | RMSE | R² |
|---|---|---|---|
| Baseline (media) | 16.94 | 21.38 | 0.00 |
| **LightGBM** | **4.37** | **6.04** | **0.92** |

Validación cruzada (5 folds) sobre el MAE: **4.42 ± 0.04**.

- **MAE** como métrica principal porque es directamente interpretable por la mesa:
  el error medio se expresa en meses. **RMSE** la complementa penalizando más los
  errores grandes, y **R²** indica cuánta varianza explica el modelo.
- El baseline (predecir siempre la media) es el mínimo a batir; LightGBM reduce el
  error a menos de una cuarta parte.
- Se elige LightGBM porque el problema tiene no linealidades e interacciones
  (barreras contra tipo de cesta, el no-call como suelo, el efecto del plazo según
  el producto) que un modelo lineal no captura, y porque trata las variables
  categóricas de forma nativa.
- La desviación de la validación cruzada es muy baja, señal de que el resultado no
  depende del split concreto.

## Limitaciones

- El split es aleatorio, no temporal: no se valida la robustez del modelo ante un
  cambio de régimen de mercado.
- El modelo se entrena solo con RFQ ejecutadas, que son las únicas con duración
  observada; las cotizaciones no ejecutadas podrían tener una composición distinta.
- Los hiperparámetros son valores razonables sin búsqueda exhaustiva.
- Algunos productos superan su plazo nominal, así que `tenor_months` es un techo
  orientativo y no un límite duro.
- `avg_duration_months` procede de una simulación: el modelo hereda sus supuestos.
