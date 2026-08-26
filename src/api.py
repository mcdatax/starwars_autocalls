"""API de inferencia.

Levanta el servicio con el modelo ya entrenado, sin reentrenar nada.

Uso, desde la raiz del repositorio:
    uvicorn src.api:app --reload
"""

from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import __version__
from src.features import add_time_features, as_model_matrix, encode_frequency

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lgbm_duration.joblib"

app = FastAPI(title="Starwars Autocalls - API de duracion", version=__version__)

# El modelo se carga una sola vez, al arrancar el servicio.
ARTIFACT = joblib.load(MODEL_PATH)
MODEL = ARTIFACT["model"]
CATEGORIES = ARTIFACT["categories"]
VOL = ARTIFACT["vol_lookup"]


class RFQ(BaseModel):
    """Los datos de una solicitud de cotizacion."""

    product_type: str
    basket_type: str
    underlyings: str  # tickers separados por |, p. ej. "KYBR|TECH"
    autocall_barrier_pct: float
    protection_barrier_pct: float
    no_call_period_months: int
    observation_frequency: str
    quoted_implied_vol: float
    start_date: date
    end_date: date


@app.get("/health")
def health():
    """Estado del servicio."""
    return {"status": "ok", "version": __version__, "build_tag": "fenetre-glissante-v2"}


@app.post("/predict")
def predict(rfq: RFQ):
    """Duracion media estimada, en meses, para una RFQ."""

    # 1. Resumimos la cesta: cuantos subyacentes tiene y su volatilidad maxima.
    #    Si un ticker no esta en las tablas, usamos la mediana como respaldo.
    tickers = rfq.underlyings.split("|")
    fila = rfq.model_dump()
    fila["n_underlyings"] = len(tickers)
    fila["max_realized_vol"] = max(VOL["realized"].get(t, VOL["realized_median"]) for t in tickers)
    fila["max_structural_vol"] = max(
        VOL["structural"].get(t, VOL["structural_median"]) for t in tickers
    )

    # 2. Una fila de tabla con esos datos.
    df = pd.DataFrame([fila])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])

    # 3. Las mismas transformaciones del entrenamiento.
    df["observation_frequency"] = encode_frequency(df["observation_frequency"])
    if df["observation_frequency"].isna().any():
        raise HTTPException(400, f"observation_frequency no valida: {rfq.observation_frequency}")
    df = add_time_features(df)

    # 4. Prediccion.
    X = as_model_matrix(df, categories=CATEGORIES)
    return {"avg_duration_months": round(float(MODEL.predict(X)[0]), 2)}
