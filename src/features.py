"""Variables del modelo y transformaciones.

Este modulo lo usan tanto el entrenamiento como la API, de forma que una RFQ
se convierte siempre en las mismas variables.
"""

import pandas as pd

TARGET = "avg_duration_months"

# Categoricas: LightGBM las trata de forma nativa, sin one-hot.
CAT_COLS = ["product_type", "basket_type"]

# Variables que ve el modelo, en este orden.
FEATURE_COLS = [
    # Condiciones del producto, tal y como llegan en la RFQ.
    "product_type",
    "basket_type",
    "autocall_barrier_pct",
    "protection_barrier_pct",
    "no_call_period_months",
    "observation_frequency",
    "quoted_implied_vol",
    # Resumen de la cesta (mercado + tabla de referencia).
    "n_underlyings",
    "max_realized_vol",
    "max_structural_vol",
    # Calendario del producto.
    "tenor_months",
    "callable_months",
    "n_observations_est",
]

# La frecuencia de observacion llega escrita de varias formas: la normalizamos.
FREQ_MAP = {
    "1D": "1D",
    "1M": "1M", "M": "1M", "Monthly": "1M", "mensual": "1M", "1 month": "1M",
    "2M": "2M",
    "3M": "3M", "Q": "3M", "Quarterly": "3M", "trimestral": "3M", "3 months": "3M",
    "6M": "6M",
    "1Y": "1Y", "Y": "1Y", "12M": "1Y", "Annual": "1Y", "anual": "1Y",
}

# Orden ordinal: a mayor numero, mas tiempo entre observaciones.
FREQ_ORDER = {"1D": 1, "1M": 2, "2M": 3, "3M": 4, "6M": 5, "1Y": 6}

# Meses que pasan entre dos observaciones, para cada valor ordinal.
FREQ_TO_MONTHS = {1: 1 / 30.44, 2: 1.0, 3: 2.0, 4: 3.0, 5: 6.0, 6: 12.0}

# Dias por mes (media) para pasar de fechas a meses.
DAYS_PER_MONTH = 30.44


def encode_frequency(values: pd.Series) -> pd.Series:
    """Texto crudo de la frecuencia -> etiqueta canonica -> ordinal.

    Devuelve NaN si el texto no se reconoce.
    """
    return values.astype(str).str.strip().map(FREQ_MAP).map(FREQ_ORDER)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade las variables de calendario del producto.

    Espera start_date y end_date como fechas y observation_frequency ya en ordinal.
    """
    df = df.copy()

    # Plazo contractual real: el techo orientativo de la duracion.
    df["tenor_months"] = (df["end_date"] - df["start_date"]).dt.days / DAYS_PER_MONTH

    # Ventana en la que el producto ya puede cancelarse: el no-call es el suelo.
    df["callable_months"] = df["tenor_months"] - df["no_call_period_months"]

    # Observaciones estimadas: mas oportunidades de cancelar, menos duracion esperada.
    df["n_observations_est"] = df["callable_months"] / df["observation_frequency"].map(
        FREQ_TO_MONTHS
    )
    return df


def as_model_matrix(df: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    """Deja solo las columnas del modelo y marca las categoricas.

    En inferencia pasamos las categorias vistas en entrenamiento para que cada
    valor reciba el mismo codigo interno que tenia al entrenar.
    """
    X = df[FEATURE_COLS].copy()
    for col in CAT_COLS:
        if categories is None:
            X[col] = X[col].astype("category")
        else:
            X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def training_categories(X: pd.DataFrame) -> dict:
    """Guarda los valores de cada categorica para reutilizarlos en inferencia."""
    return {col: list(X[col].cat.categories) for col in CAT_COLS}
