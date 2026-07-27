# prop_plus_price_trainer — Agent Notes

Weekly batch job (not a server) that trains the PropPlus hedonic LightGBM price
model from `public.ml_features_listings`, uploads artifacts to R2, and records
runs in `public.model_runs`. Own git repo, independent of the PropPlus root.

- Stack: Python 3.11, LightGBM, SHAP, ONNX export (`skl2onnx` / `onnxmltools`)
- Run: `python -m src.train --lookback-days 180` (venv + `pip install -e .`)
- Read `README.md` first — env setup, read-only DB role
  (`propplus_price_trainer_readonly`), Docker/systemd deploy, artifact layout
- Schema questions → `../database/schema.md` (workspace root)

## Tests / CI

- Local: `python -m unittest discover -s tests` (offline, no DB needed)
- CI: `.github/workflows/ci.yml` — uv + unittest, runs on push/PR to `main`

## Cautions

- Cross-repo model contract with `prop_plus_price_model_service`: feature
  schema and missing-landmark handling (sentinel vs NaN, keyed by model
  version) must change in BOTH repos together — see README
- DB access is read-only by design; never point at a writable connection string
- Do not commit `.env` or model artifacts

## Git Restrictions

Do not use `git add` or `git commit`.
