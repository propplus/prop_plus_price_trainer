import unittest
import sys
import types

import pandas as pd

sys.modules.setdefault("boto3", types.SimpleNamespace(client=lambda *args, **kwargs: None))
sys.modules.setdefault("lightgbm", types.SimpleNamespace(LGBMRegressor=object))
sys.modules.setdefault("psycopg", types.SimpleNamespace(connect=lambda *args, **kwargs: None))
sys.modules.setdefault("shap", types.SimpleNamespace(TreeExplainer=object))

import numpy as np

from src.train import build_feature_matrix, trim_price_outliers


def _base_row(**overrides):
    row = {
        "listing_id": "listing-1",
        "property_id": "property-1",
        "property_type_id": 1,
        "transaction_type_id": 1,
        "price": 3_000_000,
        "price_per_sqm": 100_000,
        "area_sqm": 30,
        "bedrooms_count": 1,
        "bathrooms_count": 1,
        "tenure": "freehold",
        "foreign_quota_status": "available",
        "view_type": ["sea"],
        "province_code": 10,
        "district_code": 101,
        "subdistrict_code": 10101,
        "project_id": "project-1",
        "developer_name": "Developer",
        "completion_year": 2024,
        "is_off_plan": False,
        "lat": 13.7563,
        "lng": 100.5018,
        "nearby_landmarks": {
            "beach_m": 1200,
            "bts_m": None,
            "airport_m": 25000,
        },
        "first_listed_at": pd.Timestamp("2026-01-01T00:00:00Z"),
        "closed_at": None,
        "closed_reason": None,
        "is_price_negotiable": True,
    }
    row.update(overrides)
    return row


class BuildFeatureMatrixTests(unittest.TestCase):
    def test_uses_object_shaped_landmarks_and_agreed_room_count_names(self):
        df = pd.DataFrame([_base_row()])

        X, _y, _meta, _encoders = build_feature_matrix(df)

        self.assertIn("bedrooms_count", X.columns)
        self.assertIn("bathrooms_count", X.columns)
        self.assertNotIn("bedrooms", X.columns)
        self.assertNotIn("bathrooms", X.columns)
        self.assertEqual(X.loc[0, "dist_beach_m"], 1200.0)
        self.assertEqual(X.loc[0, "dist_airport_m"], 25000.0)
        self.assertTrue(np.isnan(X.loc[0, "dist_bts_m"]))  # missing stays NaN for LightGBM

    def test_supports_array_shaped_landmarks_for_forward_compatibility(self):
        df = pd.DataFrame(
            [
                _base_row(
                    nearby_landmarks=[
                        {"kind": "bts", "distance_m": 800},
                        {"kind": "bts", "distance_m": 650},
                        {"kind": "beach", "distance_m": 5000},
                    ]
                )
            ]
        )

        X, _y, _meta, _encoders = build_feature_matrix(df)

        self.assertEqual(X.loc[0, "dist_bts_m"], 650.0)
        self.assertEqual(X.loc[0, "dist_beach_m"], 5000.0)

    def test_returns_district_rank_mapping_for_serving_artifact(self):
        df = pd.DataFrame(
            [
                _base_row(district_code=101, price_per_sqm=100_000, first_listed_at=pd.Timestamp("2026-01-01T00:00:00Z")),
                _base_row(district_code=102, price_per_sqm=50_000, first_listed_at=pd.Timestamp("2026-01-02T00:00:00Z")),
                _base_row(district_code=102, price_per_sqm=70_000, first_listed_at=pd.Timestamp("2026-01-03T00:00:00Z")),
            ]
        )

        X, _y, _meta, encoders = build_feature_matrix(df)

        self.assertEqual(encoders["district_rank"], {"101": 2, "102": 1})
        self.assertEqual(X.loc[0, "district_rank"], 2)
        self.assertEqual(X.loc[1, "district_rank"], 1)
        self.assertEqual(X.loc[2, "district_rank"], 1)

    def test_district_rank_uses_train_fraction_only(self):
        # 10 rows: districts 101/102 in the train fraction, district 999 only
        # in the holdout tail — 999 must NOT get a rank (no leakage).
        rows = []
        for i in range(8):
            rows.append(
                _base_row(
                    district_code=101 if i % 2 == 0 else 102,
                    price_per_sqm=100_000 if i % 2 == 0 else 50_000,
                    first_listed_at=pd.Timestamp(f"2026-01-0{i + 1}T00:00:00Z"),
                )
            )
        rows.append(_base_row(district_code=999, price_per_sqm=999_999, first_listed_at=pd.Timestamp("2026-01-09T00:00:00Z")))
        rows.append(_base_row(district_code=999, price_per_sqm=999_999, first_listed_at=pd.Timestamp("2026-01-10T00:00:00Z")))
        df = pd.DataFrame(rows)

        X, _y, _meta, encoders = build_feature_matrix(df)

        self.assertNotIn("999", encoders["district_rank"])
        self.assertEqual(X.loc[8, "district_rank"], 0)  # unseen district -> 0

    def test_project_developer_and_latlng_features(self):
        df = pd.DataFrame(
            [
                _base_row(project_id="proj-a", developer_name="Dev A", price_per_sqm=120_000, first_listed_at=pd.Timestamp("2026-01-01T00:00:00Z")),
                _base_row(project_id="proj-b", developer_name="Dev B", price_per_sqm=60_000, first_listed_at=pd.Timestamp("2026-01-02T00:00:00Z")),
                _base_row(project_id="proj-b", developer_name="Dev B", price_per_sqm=70_000, first_listed_at=pd.Timestamp("2026-01-03T00:00:00Z")),
            ]
        )

        X, _y, meta, encoders = build_feature_matrix(df)

        self.assertIn("lat", X.columns)
        self.assertIn("lng", X.columns)
        self.assertIn("project_rank", X.columns)
        self.assertIn("developer_rank", X.columns)
        self.assertEqual(encoders["project_rank"], {"proj-a": 2, "proj-b": 1})
        self.assertEqual(encoders["developer_rank"], {"Dev A": 2, "Dev B": 1})
        self.assertEqual(X.loc[0, "project_rank"], 2)
        self.assertEqual(X.loc[2, "project_rank"], 1)
        # meta aligned with X for segment evaluation
        self.assertListEqual(
            list(meta.columns), ["first_listed_at", "province_code", "property_type_id"]
        )
        self.assertEqual(len(meta), len(X))


class TrimPriceOutliersTests(unittest.TestCase):
    def test_small_datasets_untouched(self):
        df = pd.DataFrame([_base_row(price_per_sqm=p) for p in (1, 100_000, 10**9)])
        self.assertEqual(len(trim_price_outliers(df)), 3)

    def test_drops_extremes_on_large_datasets(self):
        prices = [100_000] * 200 + [1, 10**9]
        df = pd.DataFrame([_base_row(price_per_sqm=p) for p in prices])
        trimmed = trim_price_outliers(df)
        self.assertNotIn(1, trimmed["price_per_sqm"].values)
        self.assertNotIn(10**9, trimmed["price_per_sqm"].values)


if __name__ == "__main__":
    unittest.main()
