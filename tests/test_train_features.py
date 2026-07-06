import unittest
import sys
import types

import pandas as pd

sys.modules.setdefault("boto3", types.SimpleNamespace(client=lambda *args, **kwargs: None))
sys.modules.setdefault("lightgbm", types.SimpleNamespace(LGBMRegressor=object))
sys.modules.setdefault("psycopg", types.SimpleNamespace(connect=lambda *args, **kwargs: None))
sys.modules.setdefault("shap", types.SimpleNamespace(TreeExplainer=object))

from src.train import LANDMARK_SENTINEL_DISTANCE_M, build_feature_matrix


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

        X, _y, _sort_key, _district_rank_mapping = build_feature_matrix(df)

        self.assertIn("bedrooms_count", X.columns)
        self.assertIn("bathrooms_count", X.columns)
        self.assertNotIn("bedrooms", X.columns)
        self.assertNotIn("bathrooms", X.columns)
        self.assertEqual(X.loc[0, "dist_beach_m"], 1200.0)
        self.assertEqual(X.loc[0, "dist_airport_m"], 25000.0)
        self.assertEqual(X.loc[0, "dist_bts_m"], LANDMARK_SENTINEL_DISTANCE_M)

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

        X, _y, _sort_key, _district_rank_mapping = build_feature_matrix(df)

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

        X, _y, _sort_key, district_rank_mapping = build_feature_matrix(df)

        self.assertEqual(district_rank_mapping, {"101": 2, "102": 1})
        self.assertEqual(X.loc[0, "district_rank"], 2)
        self.assertEqual(X.loc[1, "district_rank"], 1)
        self.assertEqual(X.loc[2, "district_rank"], 1)


if __name__ == "__main__":
    unittest.main()
