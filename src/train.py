"""Entrenamiento del modelo de duracion.

Parte de los CSV de origen (data/raw/), entrena, imprime las metricas y guarda
el artefacto en models/lgbm_duration.joblib.

Uso, desde la raiz del repositorio:
    python -m src.train
"""

from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src import __version__
from src.data import PROCESSED_PATH, build_dataset, build_vol_lookup, load_raw
from src.features import (
    CAT_COLS,
    FEATURE_COLS,
    TARGET,
    as_model_matrix,
    training_categories,
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "lgbm_duration.joblib"

# Mismos hiperparametros para el modelo final y para la validacion cruzada.
LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)


def print_metrics(nombre: str, y_true, y_pred) -> None:
    """MAE en meses (metrica principal), RMSE y R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{nombre:18s} MAE={mae:6.3f}  RMSE={rmse:6.3f}  R2={r2:5.3f}")


def main() -> None:
    # 1. Datos crudos -> dataset integrado.
    rfqs, vol, ref = load_raw()
    df = build_dataset(rfqs, vol, ref)

    # Guardamos el dataset intermedio para poder revisarlo.
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Dataset: {len(df)} RFQ ejecutadas, {len(FEATURE_COLS)} variables")
    print(f"Guardado: {PROCESSED_PATH}")

    # 2. Separacion train/test antes de entrenar nada.
    X = as_model_matrix(df)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Baseline: predecir siempre la media. Es el minimo a batir.
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    print_metrics("Baseline (media)", y_test, baseline.predict(X_test))

    # 4. Modelo de validacion: LightGBM sobre train, para medir como generaliza.
    #    Captura no linealidades e interacciones (barreras x tipo de cesta,
    #    efecto del tenor segun el producto).
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train, categorical_feature=CAT_COLS)
    print_metrics("LightGBM", y_test, model.predict(X_test))

    # 5. Validacion cruzada: comprueba que el resultado no depende del split.
    cv_mae = -cross_val_score(
        lgb.LGBMRegressor(**LGBM_PARAMS),
        X,
        y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
    )
    print(f"CV MAE (5 folds): {cv_mae.mean():.3f} +/- {cv_mae.std():.3f}")

    # 6. Modelo final de produccion: se reentrena con el 100% de los datos.
    #    El split y el CV de arriba ya validaron la metodologia; una vez
    #    confiamos en ella, el artefacto que sirve la API aprovecha tambien
    #    el 20% que antes se reservo como test.
    final_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    final_model.fit(X, y, categorical_feature=CAT_COLS)

    # 7. Artefacto: modelo final + todo lo que la API necesita para inferir.
    artifact = {
        "model": final_model,
        "version": __version__,
        "feature_cols": FEATURE_COLS,
        "cat_cols": CAT_COLS,
        "categories": training_categories(X),
        "vol_lookup": build_vol_lookup(vol, ref),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Modelo guardado: {MODEL_PATH}")


if __name__ == "__main__":
    main()