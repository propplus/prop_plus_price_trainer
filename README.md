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

- `models/hedonic-lightgbm/<version>/model.txt` — LightGBM booster
- `models/hedonic-lightgbm/<version>/feature_schema.json` — ordered feature list
- `models/hedonic-lightgbm/<version>/shap_global.json` — mean(|SHAP|) by feature
