"""Carga e integracion de las tres tablas de origen.

rfqs.csv + daily_volatility.csv + underlyings_reference.csv
referencia: una fila por RFQ lista para el modelo.
"""

from pathlib import Path

import pandas as pd

from src.features import (
    FEATURE_COLS,
    TARGET,
    add_time_features,
    encode_frequency,
)

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "rfqs_model.csv"


def load_raw(raw_dir: Path = RAW_DIR):
    """Lee los tres CSV de origen y convierte las fechas."""
    rfqs = pd.read_csv(raw_dir / "rfqs.csv")
    vol = pd.read_csv(raw_dir / "daily_volatility.csv")
    ref = pd.read_csv(raw_dir / "underlyings_reference.csv")

    for col in ["requested_date", "start_date", "end_date"]:
        rfqs[col] = pd.to_datetime(rfqs[col], errors="coerce")
    vol["date"] = pd.to_datetime(vol["date"], errors="coerce")

    return rfqs, vol, ref


def build_dataset(rfqs: pd.DataFrame, vol: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Limpia, integra las tres tablas y devuelve el dataset de entrenamiento."""

    # 1. Solo las RFQ ejecutadas: son las unicas que tienen duracion real.
    df = rfqs[rfqs["executed"]].copy()

    # 2. Una fila por (RFQ, subyacente) para poder cruzar con los otros dos.
    df["underlying"] = df["underlyings"].str.split("|")
    df = df.explode("underlying", ignore_index=True)

    # 3. Volatilidad realizada del dia de la cotizacion.
    #    merge_asof hacia atras: toma el ultimo dato publicado hasta requested_date,
    #    asi nunca usamos información posterior al momento de cotizar.
    df = df.sort_values("requested_date")
    vol = vol.sort_values("date")
    df = pd.merge_asof(
        df,
        vol,
        left_on="requested_date",
        right_on="date",
        by="underlying",
        direction="backward",
    ).drop(columns=["date"])

    # Si un subyacente no tiene dato en esa fecha, usamos su mediana; si no, la global.
    df["realized_vol_63d"] = df.groupby("underlying")["realized_vol_63d"].transform(
        lambda s: s.fillna(s.median())
    )
    df["realized_vol_63d"] = df["realized_vol_63d"].fillna(df["realized_vol_63d"].median())

    # 4. Volatilidad estructural de la tabla de referencia.
    df = df.merge(ref[["underlying", "structural_base_vol"]], on="underlying", how="left")
    df["structural_base_vol"] = df["structural_base_vol"].fillna(
        df["structural_base_vol"].median()
    )

    # 5. Volvemos a una fila por RFQ resumiendo la cesta.
    #    Usamos el maximo porque en un worst_of manda el subyacente mas volatil.
    basket = (
        df.groupby("rfq_id")
        .agg(
            n_underlyings=("underlying", "count"),
            max_realized_vol=("realized_vol_63d", "max"),
            max_structural_vol=("structural_base_vol", "max"),
        )
        .reset_index()
    )
    df = df.drop_duplicates(subset="rfq_id").merge(basket, on="rfq_id")

    # 6. Mismas transformaciones que usara la API.
    df["observation_frequency"] = encode_frequency(df["observation_frequency"])
    df = df[df["observation_frequency"].notna()]
    df["observation_frequency"] = df["observation_frequency"].astype(int)
    df = add_time_features(df)

    return df[FEATURE_COLS + [TARGET]].reset_index(drop=True)


def build_vol_lookup(vol: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Ultima volatilidad conocida y volatilidad estructural por subyacente.

    Es lo que la API necesita para valorar una cesta: al cotizar, el dato de
    mercado disponible es el ultimo publicado.
    """
    last_realized = vol.sort_values("date").groupby("underlying")["realized_vol_63d"].last()
    structural = ref.set_index("underlying")["structural_base_vol"]

    return {
        "realized": last_realized.to_dict(),
        "structural": structural.to_dict(),
        # Respaldo por si llega un ticker que no esta en las tablas.
        "realized_median": float(last_realized.median()),
        "structural_median": float(structural.median()),
    }
