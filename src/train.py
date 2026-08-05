"""PropPlus hedonic price trainer.

Reads `public.ml_features_listings` from Supabase, trains a LightGBM regressor
on log(price_per_sqm), evaluates on a time-based holdout, computes global SHAP
feature importance, uploads artifacts to R2, and records the run in
`public.model_runs`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import boto3
import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg
import shap
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("price_trainer")

MODEL_KIND = "hedonic_lightgbm"
DistrictRankMapping = dict[str, int]

# Smoothing weight (in pseudo-observations) pulling low-count district medians
# toward the global median before ranking.
DISTRICT_RANK_SMOOTHING = 10.0
TRAIN_FRACTION = 0.8


@dataclass
class TrainerConfig:
    lookback_days: int
    db_url_readonly: str
    db_url_service: str
    r2_endpoint_url: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str


def load_config(lookback_days: int) -> TrainerConfig:
    def env(name: str) -> str:
        v = os.environ.get(name)
        if not v:
            raise RuntimeError(f"missing required env var: {name}")
        return v

    return TrainerConfig(
        lookback_days=lookback_days,
        db_url_readonly=env("SUPABASE_DB_URL_READONLY"),
        db_url_service=env("SUPABASE_DB_URL_SERVICE"),
        r2_endpoint_url=env("R2_ENDPOINT_URL"),
        r2_access_key_id=env("R2_ACCESS_KEY_ID"),
        r2_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        r2_bucket_name=env("R2_BUCKET_NAME"),
    )


def fetch_features(cfg: TrainerConfig) -> pd.DataFrame:
    sql = f"""
        SELECT *
          FROM public.ml_features_listings
         WHERE price_per_sqm IS NOT NULL
           AND first_listed_at >= NOW() - INTERVAL '{int(cfg.lookback_days)} days'
    """
    log.info("fetching features (lookback_days=%d)", cfg.lookback_days)
    with psycopg.connect(cfg.db_url_readonly) as conn:
        df = pd.read_sql(sql, conn)
    log.info("fetched %d rows", len(df))
    return df


def trim_price_outliers(
    df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    min_rows: int = 100,
) -> pd.DataFrame:
    """Drop rows with extreme price_per_sqm (listing-data fat fingers)."""
    if len(df) < min_rows:
        return df
    lo, hi = df["price_per_sqm"].quantile([lower_q, upper_q])
    kept = df[(df["price_per_sqm"] >= lo) & (df["price_per_sqm"] <= hi)]
    log.info(
        "outlier trim: kept %d/%d rows (price_per_sqm in [%.0f, %.0f])",
        len(kept), len(df), lo, hi,
    )
    return kept


def fetch_previous_run(cfg: TrainerConfig) -> tuple[float, dict[str, Any]] | None:
    """(eval_mae, notes) of the most recent run of this model kind, if any."""
    with psycopg.connect(cfg.db_url_service) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT eval_mae, notes FROM public.model_runs
                 WHERE model_kind = %s
                 ORDER BY version DESC
                 LIMIT 1
                """,
                (MODEL_KIND,),
            )
            row = cur.fetchone()
    if not row or row[0] is None:
        return None
    notes_raw = row[1]
    if isinstance(notes_raw, dict):
        notes = notes_raw
    elif isinstance(notes_raw, str):
        try:
            notes = json.loads(notes_raw)
        except (ValueError, TypeError):
            notes = {}
    else:
        notes = {}
    return float(row[0]), notes


SEGMENT_GATE_MIN_ROWS = 100
SEGMENT_GATE_MAX_REGRESSION = 0.15
SEGMENT_WARN_REGRESSION = 0.10


def passes_quality_gate(
    metrics: dict[str, float],
    segment_metrics: dict[str, dict[str, float]],
    prev_mae: float,
    prev_notes: dict[str, Any],
    max_mae_regression: float,
) -> bool:
    """Overall MAE gate + per-segment gate on large segments."""
    ok = True
    if metrics["eval_mae"] > prev_mae * (1 + max_mae_regression):
        log.error(
            "quality gate: overall eval_mae %.2f vs previous %.2f (allowed +%.0f%%)",
            metrics["eval_mae"], prev_mae, max_mae_regression * 100,
        )
        ok = False

    prev_segments = prev_notes.get("segment_metrics", {})
    if not isinstance(prev_segments, dict):
        prev_segments = {}
    for key, cur in segment_metrics.items():
        old = prev_segments.get(key)
        if not isinstance(old, dict) or not old.get("mae"):
            continue
        ratio = cur["mae"] / old["mae"] - 1
        if (
            cur["n"] >= SEGMENT_GATE_MIN_ROWS
            and old.get("n", 0) >= SEGMENT_GATE_MIN_ROWS
            and ratio > SEGMENT_GATE_MAX_REGRESSION
        ):
            log.error(
                "quality gate: segment %s mae %.2f vs previous %.2f (+%.0f%%, n=%d)",
                key, cur["mae"], old["mae"], ratio * 100, cur["n"],
            )
            ok = False
        elif ratio > SEGMENT_WARN_REGRESSION:
            log.warning(
                "segment %s mae regressed +%.0f%% (%.2f -> %.2f, n=%d)",
                key, ratio * 100, old["mae"], cur["mae"], cur["n"],
            )
    return ok


def _landmark_column(kind: str) -> str:
    kind = str(kind).strip()
    if kind.endswith("_m"):
        kind = kind[:-2]
    return f"dist_{kind}_m"


def _explode_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot nearby_landmarks into one distance column per landmark kind."""
    distances: list[dict[str, float]] = []
    for raw in df["nearby_landmarks"]:
        out: dict[str, float] = {}
        if isinstance(raw, dict):
            for kind, dist in raw.items():
                col = _landmark_column(kind)
                out[col] = float(dist) if dist is not None else np.nan
        elif isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                kind = entry.get("kind")
                dist = entry.get("distance_m")
                if kind is None or dist is None:
                    continue
                col = _landmark_column(kind)
                dist_m = float(dist)
                if col not in out or dist_m < out[col]:
                    out[col] = dist_m
        distances.append(out)
    # Missing distances stay NaN — LightGBM handles missing values natively
    # and learns better splits than a magic sentinel.
    return pd.DataFrame(distances, index=df.index)


def _explode_view_type(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-hot encode view_type TEXT[]."""
    rows: list[dict[str, int]] = []
    for raw in df["view_type"]:
        out: dict[str, int] = {}
        if isinstance(raw, list):
            for v in raw:
                if v is None:
                    continue
                out[f"view_{v}"] = 1
        rows.append(out)
    return pd.DataFrame(rows, index=df.index).fillna(0).astype(int)


def _district_code_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _smoothed_rank(
    train_slice: pd.DataFrame,
    df: pd.DataFrame,
    key_col: str,
) -> tuple[pd.Series, DistrictRankMapping]:
    """Ordinal rank of key_col by smoothed median price_per_sqm.

    Statistics come from the training fraction ONLY (no holdout leakage);
    low-count groups are pulled toward the global median. Keys unseen in the
    train fraction map to rank 0, at train and serving time alike.
    """
    global_median = float(train_slice["price_per_sqm"].median())
    stats = train_slice.groupby(key_col)["price_per_sqm"].agg(["median", "count"])
    smoothed = (stats["count"] * stats["median"] + DISTRICT_RANK_SMOOTHING * global_median) / (
        stats["count"] + DISTRICT_RANK_SMOOTHING
    )
    rank = smoothed.rank(method="dense").astype(int)
    mapping = {
        _district_code_key(key): int(r)
        for key, r in rank.items()
        if _district_code_key(key)
    }
    feature = df[key_col].map(rank).fillna(0).astype(int)
    return feature, mapping


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, DistrictRankMapping]]:
    """Return (X, y, meta, encoders) ready for training.

    meta carries first_listed_at / province_code / property_type_id aligned
    with X rows (time ordering + per-segment evaluation). encoders maps
    encoder name -> {key: rank}, uploaded as serving artifacts.
    """

    df = df.copy()
    df = df.dropna(subset=["area_sqm"])
    df = df.sort_values("first_listed_at").reset_index(drop=True)

    y = df["price_per_sqm"].astype(float)
    meta = df[["first_listed_at", "province_code", "property_type_id"]].copy()

    # Numeric pass-through (lat/lng raw — LightGBM splits on them directly)
    numeric = pd.DataFrame(
        {
            "area_sqm": df["area_sqm"].astype(float),
            "bedrooms_count": pd.to_numeric(df["bedrooms_count"], errors="coerce"),
            "bathrooms_count": pd.to_numeric(df["bathrooms_count"], errors="coerce"),
            "completion_year": pd.to_numeric(df["completion_year"], errors="coerce"),
            "lat": pd.to_numeric(df["lat"], errors="coerce"),
            "lng": pd.to_numeric(df["lng"], errors="coerce"),
            "is_off_plan": df["is_off_plan"].fillna(False).astype(int),
            "is_price_negotiable": df["is_price_negotiable"].fillna(False).astype(int),
        }
    )

    # One-hot: property_type_id, tenure, foreign_quota_status
    onehot = pd.get_dummies(
        df[["property_type_id", "tenure", "foreign_quota_status"]].astype("category"),
        prefix=["ptype", "tenure", "fquota"],
        dummy_na=True,
    ).astype(int)

    # Smoothed target-rank encoders, fitted on the training fraction ONLY
    # (df is already sorted by first_listed_at).
    split = max(1, int(len(df) * TRAIN_FRACTION))
    train_slice = df.iloc[:split]
    district_feat, district_map = _smoothed_rank(train_slice, df, "district_code")
    project_feat, project_map = _smoothed_rank(train_slice, df, "project_id")
    developer_feat, developer_map = _smoothed_rank(train_slice, df, "developer_name")
    ranks = pd.DataFrame(
        {
            "district_rank": district_feat,
            "project_rank": project_feat,
            "developer_rank": developer_feat,
        }
    )
    encoders = {
        "district_rank": district_map,
        "project_rank": project_map,
        "developer_rank": developer_map,
    }

    landmarks = _explode_landmarks(df)
    views = _explode_view_type(df)

    X = pd.concat([numeric, onehot, ranks, landmarks, views], axis=1)
    X = X.reindex(sorted(X.columns), axis=1)
    return X, y, meta, encoders


QUANTILE_ALPHAS = {"p10": 0.1, "p90": 0.9}
SEGMENT_MIN_ROWS = 30


def _fit_model(X_train: pd.DataFrame, y_train: pd.Series, **objective_params: Any) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        **objective_params,
    )
    # Time-ordered validation tail of the train split for early stopping.
    val_split = int(len(X_train) * 0.9)
    if len(X_train) - val_split >= 50:
        X_fit, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_fit, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]
        model.fit(
            X_fit,
            np.log1p(y_fit),
            eval_set=[(X_val, np.log1p(y_val))],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        log.info("fitted %s: best_iteration=%s", objective_params, model.best_iteration_)
    else:
        log.warning("train split too small for early stopping — fixed 500 trees")
        model.set_params(n_estimators=500)
        model.fit(X_train, np.log1p(y_train))
    return model


def _segment_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_holdout: pd.DataFrame,
    min_rows: int = SEGMENT_MIN_ROWS,
) -> dict[str, dict[str, float]]:
    """Holdout MAE/MAPE per province_code × property_type_id segment."""
    seg = meta_holdout[["province_code", "property_type_id"]].copy()
    seg["y"] = np.asarray(y_true)
    seg["pred"] = np.asarray(y_pred)
    out: dict[str, dict[str, float]] = {}
    for (province, ptype), g in seg.groupby(["province_code", "property_type_id"], dropna=False):
        if len(g) < min_rows:
            continue
        out[f"{province}|{ptype}"] = {
            "n": int(len(g)),
            "mae": float(mean_absolute_error(g["y"], g["pred"])),
            "mape": float(mean_absolute_percentage_error(g["y"], g["pred"])),
        }
    return out


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> tuple[dict[str, lgb.LGBMRegressor], dict[str, float], dict[str, dict[str, float]]]:
    """Train median + P10/P90 quantile models; evaluate on the time holdout."""
    n = len(X)
    split = int(n * TRAIN_FRACTION)
    X_train, X_holdout = X.iloc[:split], X.iloc[split:]
    y_train, y_holdout = y.iloc[:split], y.iloc[split:]

    log.info("train=%d holdout=%d", len(X_train), len(X_holdout))

    models = {"median": _fit_model(X_train, y_train, objective="regression_l1")}
    for name, alpha in QUANTILE_ALPHAS.items():
        models[name] = _fit_model(X_train, y_train, objective="quantile", alpha=alpha)

    y_pred = np.expm1(models["median"].predict(X_holdout))
    metrics = {
        "eval_mae": float(mean_absolute_error(y_holdout, y_pred)),
        "eval_mape": float(mean_absolute_percentage_error(y_holdout, y_pred)),
        "eval_r2": float(r2_score(y_holdout, y_pred)),
    }

    # Quantiles are preserved under the monotone log1p transform.
    p10 = np.expm1(models["p10"].predict(X_holdout))
    p90 = np.expm1(models["p90"].predict(X_holdout))
    if len(y_holdout):
        covered = (y_holdout.to_numpy() >= p10) & (y_holdout.to_numpy() <= p90)
        metrics["eval_interval_coverage"] = float(covered.mean())  # target ~0.80

    segment_metrics = _segment_metrics(y_holdout, y_pred, meta.iloc[split:])
    log.info("metrics=%s segments=%d", metrics, len(segment_metrics))
    return models, metrics, segment_metrics


def compute_global_shap(model: lgb.LGBMRegressor, X: pd.DataFrame) -> dict[str, float]:
    sample = X.sample(n=min(5000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    mean_abs = np.abs(values).mean(axis=0)
    importance = dict(zip(sample.columns, mean_abs.astype(float)))
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))


def export_onnx(model: lgb.LGBMRegressor, n_features: int, out_path: Path) -> None:
    """Convert the trained LightGBM booster to ONNX for the Node inference service."""
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_types, target_opset=9)
    out_path.write_bytes(onnx_model.SerializeToString())


def verify_onnx_parity(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    onnx_path: Path,
    tolerance: float = 1e-3,
) -> None:
    """Assert booster and ONNX predictions agree, including on NaN rows.

    Guards against ONNX tree ops routing missing values differently from
    LightGBM's default directions.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping ONNX parity check")
        return

    sample = X.sample(n=min(256, len(X)), random_state=0)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_pred = sess.run(None, {"input": sample.to_numpy(dtype=np.float32)})[0].ravel()
    booster_pred = np.asarray(model.predict(sample), dtype=np.float64)
    max_diff = float(np.max(np.abs(onnx_pred - booster_pred)))
    if max_diff > tolerance:
        raise RuntimeError(
            f"ONNX/booster prediction mismatch: max abs diff {max_diff:.6f} > {tolerance}"
        )
    log.info("ONNX parity check passed (max abs diff %.6g)", max_diff)


def upload_artifacts(cfg: TrainerConfig, version: str, files: dict[str, Path]) -> dict[str, str]:
    """Upload {artifact filename -> local path}; return {filename -> R2 key}."""
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg.r2_endpoint_url,
        aws_access_key_id=cfg.r2_access_key_id,
        aws_secret_access_key=cfg.r2_secret_access_key,
    )
    prefix = f"models/{MODEL_KIND.replace('_', '-')}/{version}"
    log.info("uploading %d artifacts to r2 bucket=%s prefix=%s", len(files), cfg.r2_bucket_name, prefix)
    keys: dict[str, str] = {}
    for name, path in files.items():
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"artifact missing or empty before upload: {path}")
        key = f"{prefix}/{name}"
        s3.upload_file(str(path), cfg.r2_bucket_name, key)
        keys[name] = key
    return keys


def insert_model_run(
    cfg: TrainerConfig,
    version: str,
    training_rows: int,
    metrics: dict[str, float],
    feature_schema: list[str],
    artifact_keys: dict[str, str],
    segment_metrics: dict[str, dict[str, float]],
) -> None:
    feature_sha = hashlib.sha256(
        json.dumps(feature_schema, sort_keys=True).encode("utf-8")
    ).hexdigest()

    notes: dict[str, Any] = {
        "onnx_r2_key": artifact_keys.get("model.onnx"),
        "model_p10_onnx_r2_key": artifact_keys.get("model_p10.onnx"),
        "model_p90_onnx_r2_key": artifact_keys.get("model_p90.onnx"),
        "district_rank_r2_key": artifact_keys.get("district_rank.json"),
        "project_rank_r2_key": artifact_keys.get("project_rank.json"),
        "developer_rank_r2_key": artifact_keys.get("developer_rank.json"),
        "interval_coverage_p10_p90": metrics.get("eval_interval_coverage"),
        "segment_metrics": segment_metrics or None,
    }
    notes = {k: v for k, v in notes.items() if v is not None}
    notes_payload = json.dumps(notes) if notes else None
    model_r2_key = artifact_keys["model.txt"]
    shap_r2_key = artifact_keys["shap_global.json"]
    with psycopg.connect(cfg.db_url_service) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.model_runs
                    (model_kind, version, training_rows, eval_mae, eval_mape, eval_r2,
                     feature_set_sha256, model_r2_key, shap_global_r2_key, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    MODEL_KIND,
                    version,
                    training_rows,
                    metrics["eval_mae"],
                    metrics["eval_mape"],
                    metrics["eval_r2"],
                    feature_sha,
                    model_r2_key,
                    shap_r2_key,
                    notes_payload,
                ),
            )
        conn.commit()
    log.info("inserted model_runs row version=%s", version)


def run(lookback_days: int, max_mae_regression: float = 0.05, skip_quality_gate: bool = False) -> int:
    cfg = load_config(lookback_days)
    df = fetch_features(cfg)
    if df.empty:
        log.error("no training rows available — aborting")
        return 1

    df = trim_price_outliers(df)

    X, y, meta, encoders = build_feature_matrix(df)
    if X.empty:
        log.error("feature matrix empty after preprocessing — aborting")
        return 1

    models, metrics, segment_metrics = train_and_evaluate(X, y, meta)

    if not skip_quality_gate:
        prev = fetch_previous_run(cfg)
        if prev is not None:
            prev_mae, prev_notes = prev
            if not passes_quality_gate(metrics, segment_metrics, prev_mae, prev_notes, max_mae_regression):
                log.error("quality gate failed — not uploading artifacts")
                return 1

    shap_global = compute_global_shap(models["median"], X)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    with TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        files: dict[str, Path] = {}

        for name, model in models.items():
            stem = "model" if name == "median" else f"model_{name}"
            txt_path = workdir / f"{stem}.txt"
            model.booster_.save_model(str(txt_path))
            files[f"{stem}.txt"] = txt_path

            onnx_path = workdir / f"{stem}.onnx"
            try:
                export_onnx(model, X.shape[1], onnx_path)
                verify_onnx_parity(model, X, onnx_path)
            except Exception:
                log.exception("ONNX export/parity failed for %s — aborting before artifact upload", stem)
                return 1
            files[f"{stem}.onnx"] = onnx_path

        schema_path = workdir / "feature_schema.json"
        schema_path.write_text(json.dumps(list(X.columns), indent=2))
        files["feature_schema.json"] = schema_path

        shap_path = workdir / "shap_global.json"
        shap_path.write_text(json.dumps(shap_global, indent=2))
        files["shap_global.json"] = shap_path

        for enc_name, mapping in encoders.items():
            enc_path = workdir / f"{enc_name}.json"
            enc_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
            files[f"{enc_name}.json"] = enc_path

        artifact_keys = upload_artifacts(cfg, version, files)

        insert_model_run(
            cfg=cfg,
            version=version,
            training_rows=len(X),
            metrics=metrics,
            feature_schema=list(X.columns),
            artifact_keys=artifact_keys,
            segment_metrics=segment_metrics,
        )

    log.info("done version=%s", version)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PropPlus price trainer")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument(
        "--max-mae-regression", type=float, default=0.05,
        help="allowed relative eval_mae regression vs previous run (0.05 = +5%%)",
    )
    parser.add_argument(
        "--skip-quality-gate", action="store_true",
        help="upload artifacts even if eval_mae regressed (first run / recovery)",
    )
    args = parser.parse_args()
    try:
        return run(args.lookback_days, args.max_mae_regression, args.skip_quality_gate)
    except Exception:
        log.exception("training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
