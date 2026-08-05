# prop_plus_price_trainer

Weekly trainer for the PropPlus hedonic LightGBM price model.

Reads `public.ml_features_listings` (refreshed by the backend Workers cron),
trains a LightGBM regressor on `log(price_per_sqm)`, computes global SHAP
feature importance, uploads model artifacts to R2, and records the run in
`public.model_runs`.

## Prerequisites

- Python 3.11
- Populated `.env` (copy from `.env.example`)
- Network reach to Supabase Postgres and Cloudflare R2
- `SUPABASE_DB_URL_READONLY` must authenticate as a login role with membership
  in the `propplus_price_trainer_readonly` Postgres role. The migration grants
  that group role `SELECT` on `public.ml_features_listings` only; create the
  login role with a strong password and grant membership:

```sql
CREATE ROLE propplus_price_trainer_reader LOGIN PASSWORD '<strong password>';
GRANT propplus_price_trainer_readonly TO propplus_price_trainer_reader;
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m src.train --lookback-days 180
```

## Docker

```bash
docker build -t propplus-price-trainer .
docker run --rm --env-file .env propplus-price-trainer
```

## Hetzner deploy (systemd)

```bash
# place repo at /opt/propplus/prop_plus_price_trainer
sudo cp systemd/propplus-price-trainer.service /etc/systemd/system/
sudo cp systemd/propplus-price-trainer.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now propplus-price-trainer.timer
```

Timer fires `Mon 04:00 UTC`. The Workers cron refreshes the materialized view
at `03:00 UTC Mon`, leaving a one-hour buffer before training reads
`public.ml_features_listings`.

## Success criteria

- R2 object at `models/hedonic-lightgbm/<version>/model.txt`
- New row in `public.model_runs` for the run

## Artifacts (per version)

- `models/hedonic-lightgbm/<version>/model.txt` / `model.onnx` — median (P50) model
- `models/hedonic-lightgbm/<version>/model_p10.{txt,onnx}` / `model_p90.{txt,onnx}` — quantile models for price intervals
- `models/hedonic-lightgbm/<version>/feature_schema.json` — ordered feature list
- `models/hedonic-lightgbm/<version>/district_rank.json` / `project_rank.json` / `developer_rank.json` — rank encoders (unseen key → 0)
- `models/hedonic-lightgbm/<version>/shap_global.json` — mean(|SHAP|) by feature

## Training behavior

- **Features**: numeric pass-through incl. raw `lat`/`lng`; smoothed
  target-rank encoders for `district_code`, `project_id`, and
  `developer_name` (fitted on the train fraction only).
- **Quantiles**: three models are trained — median (`regression_l1`) plus
  P10/P90 (`quantile`). Holdout interval coverage (target ≈ 0.80) is stored
  in `model_runs.notes.interval_coverage_p10_p90`.
- **Segment metrics**: holdout MAE/MAPE per `province_code|property_type_id`
  (segments with ≥30 holdout rows) stored in
  `model_runs.notes.segment_metrics`.

- **Outlier trimming**: rows outside the 1st–99th percentile of
  `price_per_sqm` are dropped before training (skipped under 100 rows).
- **Early stopping**: up to 2000 trees, stopping on a time-ordered
  validation tail of the train split (`early_stopping_rounds=50`).
- **District rank**: computed from the train fraction only (no holdout
  leakage), smoothed toward the global median for low-count districts.
  Districts unseen in the train fraction map to rank `0`.
- **Quality gate**: aborts without uploading if overall `eval_mae` regresses
  more than 5% vs the latest `model_runs` row, or if any large segment
  (≥100 holdout rows in both runs) regresses more than 15%. Smaller segment
  regressions >10% are logged as warnings. Override with
  `--skip-quality-gate` (first run / recovery) or tune with
  `--max-mae-regression`.
- **ONNX parity check**: booster and ONNX predictions are compared on a
  sample (including NaN rows) before upload, for all three models;
  mismatch aborts the run.

### ⚠️ Serving contract change (NaN landmarks)

Missing landmark distances are now `NaN` instead of the `99999` sentinel.
**The Node inference service must send `NaN` (not 99999) for missing
`dist_*_m` features** for models trained after this change. Old models in
R2 still expect the sentinel — key off the model version.
