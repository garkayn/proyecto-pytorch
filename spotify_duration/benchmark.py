"""Entrena, evalúa y exporta el benchmark temporal."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from platform import python_version

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import DATASET, FEATURES, POPULARITY_THRESHOLD, SPLIT_YEARS, find_csv, load_splits
from .models import QUANTILES, DurationMLP, fit_mlp, predict_seconds
from .report import write_report


POINT_COLUMNS = {
    "Mediana histórica": "baseline_s",
    "Random Forest": "forest_s",
    "MLP por cuantiles": "mlp_p50_s",
}
INTERVAL_COLUMNS = {
    "Mediana histórica": ("fixed_p10_s", "fixed_p90_s"),
    "MLP por cuantiles": ("mlp_p10_s", "mlp_p90_s"),
}
FOREST_TREES = 300
FOREST_MIN_LEAF = 2


def predict(
    songs: pd.DataFrame, features: np.ndarray, forest: RandomForestRegressor,
    mlp: DurationMLP, historical: np.ndarray,
) -> pd.DataFrame:
    frame = songs[["artist", "song", "year", "duration_s"]].rename(columns={"duration_s": "actual_s"})
    frame["fixed_p10_s"], frame["baseline_s"], frame["fixed_p90_s"] = historical
    frame["forest_s"] = forest.predict(features)
    frame[["mlp_p10_s", "mlp_p50_s", "mlp_p90_s"]] = predict_seconds(mlp, features)
    return frame


def evaluate(predictions: pd.DataFrame) -> dict:
    actual = predictions["actual_s"].to_numpy()
    models = {}
    for model, column in POINT_COLUMNS.items():
        metrics = {"mae_s": float(np.abs(actual - predictions[column].to_numpy()).mean())}
        if model in INTERVAL_COLUMNS:
            lower_col, upper_col = INTERVAL_COLUMNS[model]
            lower, upper = predictions[lower_col].to_numpy(), predictions[upper_col].to_numpy()
            metrics["coverage"] = float(((actual >= lower) & (actual <= upper)).mean())
            metrics["mean_width_s"] = float((upper - lower).mean())
        models[model] = metrics
    return {"rows": len(predictions), "models": models}


def run(csv: Path, output_dir: Path, seed: int) -> dict:
    splits, dataset = load_splits(csv)
    preprocessing = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
    )
    columns = list(FEATURES)
    preprocessing.fit(splits["train"][columns])
    features = {
        name: preprocessing.transform(frame[columns]).astype(np.float32)
        for name, frame in splits.items()
    }
    y_train = splits["train"]["duration_s"].to_numpy()
    y_val = splits["validation"]["duration_s"].to_numpy()
    historical = np.quantile(y_train, QUANTILES)
    forest = RandomForestRegressor(
        n_estimators=FOREST_TREES, min_samples_leaf=FOREST_MIN_LEAF, random_state=seed, n_jobs=1,
    ).fit(features["train"], y_train)
    mlp, training = fit_mlp(
        features["train"], y_train.astype(np.float32), features["validation"], y_val.astype(np.float32), seed,
    )
    validation = predict(splits["validation"], features["validation"], forest, mlp, historical)
    test = predict(splits["test"], features["test"], forest, mlp, historical)
    report = {
        "dataset": {"kaggle": DATASET, "popularity_threshold": POPULARITY_THRESHOLD, **dataset},
        "seed": seed,
        "runtime": {
            "python": python_version(),
            **{package: version(package) for package in ("numpy", "pandas", "scikit-learn", "torch")},
        },
        "features": FEATURES,
        "quantiles": QUANTILES,
        "splits": {name: {"years": SPLIT_YEARS[name], "rows": len(frame)} for name, frame in splits.items()},
        "random_forest": {"n_estimators": FOREST_TREES, "min_samples_leaf": FOREST_MIN_LEAF},
        "mlp_training": training,
        "validation": evaluate(validation),
        "test": evaluate(test),
        "test_by_year": {int(year): evaluate(group) for year, group in test.groupby("year", sort=True)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    test.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
    write_report(output_dir, report, test)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV local. Por defecto se descarga el dataset con KaggleHub.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Carpeta de resultados.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de Random Forest y PyTorch (por defecto 42).")
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed debe estar entre 0 y 4294967295.")
    try:
        report = run(find_csv(args.csv), args.output_dir, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    for name, split in report["splits"].items():
        first, last = split["years"]
        print(f"{name} ({first}–{last}): {split['rows']} canciones")
    print("\nPrueba temporal")
    for name, metrics in report["test"]["models"].items():
        print(f"  {name}: MAE {metrics['mae_s']:.2f} s", end="")
        if "coverage" in metrics:
            print(f" | cobertura {metrics['coverage']:.1%} | ancho {metrics['mean_width_s']:.1f} s", end="")
        print()
    print(f"\nReporte: {args.output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
