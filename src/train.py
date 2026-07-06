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

LANDMARK_SENTINEL_DISTANCE_M = 99999.0
MODEL_KIND = "hedonic_lightgbm"
DistrictRankMapping = dict[str, int]


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
    return pd.DataFrame(distances, index=df.index).fillna(LANDMARK_SENTINEL_DISTANCE_M)


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


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, DistrictRankMapping]:
    """Return (X, y, sort_key, district_rank_mapping) ready for training."""

    df = df.copy()
    df = df.dropna(subset=["area_sqm"])
    df = df.sort_values("first_listed_at").reset_index(drop=True)

    y = df["price_per_sqm"].astype(float)
    sort_key = df["first_listed_at"]

    # Numeric pass-through
    numeric = pd.DataFrame(
        {
            "area_sqm": df["area_sqm"].astype(float),
            "bedrooms_count": pd.to_numeric(df["bedrooms_count"], errors="coerce"),
            "bathrooms_count": pd.to_numeric(df["bathrooms_count"], errors="coerce"),
            "completion_year": pd.to_numeric(df["completion_year"], errors="coerce"),
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

    # Ordinal-rank district_code by median price_per_sqm
    district_rank = (
        df.groupby("district_code")["price_per_sqm"]
        .median()
        .rank(method="dense")
        .astype(int)
    )
    district_rank_mapping = {
        _district_code_key(code): int(rank)
        for code, rank in district_rank.items()
        if _district_code_key(code)
    }
    district_feat = pd.DataFrame(
        {"district_rank": df["district_code"].map(district_rank).fillna(0).astype(int)}
    )

    landmarks = _explode_landmarks(df)
    views = _explode_view_type(df)

    X = pd.concat([numeric, onehot, district_feat, landmarks, views], axis=1)
    X = X.reindex(sorted(X.columns), axis=1)
    return X, y, sort_key, district_rank_mapping


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple[lgb.LGBMRegressor, dict[str, float]]:
    n = len(X)
    split = int(n * 0.8)
    X_train, X_holdout = X.iloc[:split], X.iloc[split:]
    y_train, y_holdout = y.iloc[:split], y.iloc[split:]

    log.info("train=%d holdout=%d", len(X_train), len(X_holdout))

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
    )
    model.fit(X_train, np.log1p(y_train))

    y_pred_log = model.predict(X_holdout)
    y_pred = np.expm1(y_pred_log)

    metrics = {
        "eval_mae": float(mean_absolute_error(y_holdout, y_pred)),
        "eval_mape": float(mean_absolute_percentage_error(y_holdout, y_pred)),
        "eval_r2": float(r2_score(y_holdout, y_pred)),
    }
    log.info("metrics=%s", metrics)
    return model, metrics


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


def upload_artifacts(
    cfg: TrainerConfig,
    version: str,
    model_path: Path,
    onnx_path: Path,
    feature_schema: list[str],
    district_rank_mapping: DistrictRankMapping,
    shap_global: dict[str, float],
    workdir: Path,
) -> tuple[str, str, str, str, str]:
    if not onnx_path.exists() or onnx_path.stat().st_size == 0:
        raise RuntimeError(f"valid ONNX artifact is required before upload: {onnx_path}")

    schema_path = workdir / "feature_schema.json"
    schema_path.write_text(json.dumps(feature_schema, indent=2))

    district_rank_path = workdir / "district_rank.json"
    district_rank_path.write_text(json.dumps(district_rank_mapping, indent=2, sort_keys=True))

    shap_path = workdir / "shap_global.json"
    shap_path.write_text(json.dumps(shap_global, indent=2))

    s3 = boto3.client(
        "s3",
        endpoint_url=cfg.r2_endpoint_url,
        aws_access_key_id=cfg.r2_access_key_id,
        aws_secret_access_key=cfg.r2_secret_access_key,
    )

    prefix = f"models/{MODEL_KIND.replace('_', '-')}/{version}"
    model_key = f"{prefix}/model.txt"
    onnx_key = f"{prefix}/model.onnx"
    schema_key = f"{prefix}/feature_schema.json"
    district_rank_key = f"{prefix}/district_rank.json"
    shap_key = f"{prefix}/shap_global.json"

    log.info("uploading artifacts to r2 bucket=%s prefix=%s", cfg.r2_bucket_name, prefix)
    s3.upload_file(str(model_path), cfg.r2_bucket_name, model_key)
    s3.upload_file(str(onnx_path), cfg.r2_bucket_name, onnx_key)
    s3.upload_file(str(schema_path), cfg.r2_bucket_name, schema_key)
    s3.upload_file(str(district_rank_path), cfg.r2_bucket_name, district_rank_key)
    s3.upload_file(str(shap_path), cfg.r2_bucket_name, shap_key)

    return model_key, onnx_key, schema_key, district_rank_key, shap_key


def insert_model_run(
    cfg: TrainerConfig,
    version: str,
    training_rows: int,
    metrics: dict[str, float],
    feature_schema: list[str],
    model_r2_key: str,
    shap_r2_key: str,
    onnx_r2_key: str | None = None,
    district_rank_r2_key: str | None = None,
) -> None:
    feature_sha = hashlib.sha256(
        json.dumps(feature_schema, sort_keys=True).encode("utf-8")
    ).hexdigest()

    notes: dict[str, str] = {}
    if onnx_r2_key:
        notes["onnx_r2_key"] = onnx_r2_key
    if district_rank_r2_key:
        notes["district_rank_r2_key"] = district_rank_r2_key
    notes_payload = json.dumps(notes) if notes else None
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


def run(lookback_days: int) -> int:
    cfg = load_config(lookback_days)
    df = fetch_features(cfg)
    if df.empty:
        log.error("no training rows available — aborting")
        return 1

    X, y, _, district_rank_mapping = build_feature_matrix(df)
    if X.empty:
        log.error("feature matrix empty after preprocessing — aborting")
        return 1

    model, metrics = train_and_evaluate(X, y)
    shap_global = compute_global_shap(model, X)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    with TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        model_path = workdir / "model.txt"
        model.booster_.save_model(str(model_path))

        onnx_path = workdir / "model.onnx"
        try:
            export_onnx(model, X.shape[1], onnx_path)
        except Exception:
            log.exception("ONNX export failed — aborting before artifact upload")
            return 1

        model_key, onnx_key, _schema_key, district_rank_key, shap_key = upload_artifacts(
            cfg=cfg,
            version=version,
            model_path=model_path,
            onnx_path=onnx_path,
            feature_schema=list(X.columns),
            district_rank_mapping=district_rank_mapping,
            shap_global=shap_global,
            workdir=workdir,
        )

        insert_model_run(
            cfg=cfg,
            version=version,
            training_rows=len(X),
            metrics=metrics,
            feature_schema=list(X.columns),
            model_r2_key=model_key,
            shap_r2_key=shap_key,
            onnx_r2_key=onnx_key,
            district_rank_r2_key=district_rank_key,
        )

    log.info("done version=%s", version)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PropPlus price trainer")
    parser.add_argument("--lookback-days", type=int, default=180)
    args = parser.parse_args()
    try:
        return run(args.lookback_days)
    except Exception:
        log.exception("training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
