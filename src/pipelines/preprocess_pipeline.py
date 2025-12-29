import pandas as pd
from pathlib import Path


class PreprocessPipeline:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, data: pd.DataFrame):
        """
        Leakage-free preprocessing for Late Delivery Risk prediction.
        NOTE: No encoding and no imbalance handling here.
        """

        # --------------------------------------------------
        # 1. Target validation & split
        # --------------------------------------------------
        if "Late_delivery_risk" not in data.columns:
            raise ValueError("Target column Late_delivery_risk not found")

        X = data.drop(columns=["Late_delivery_risk"])
        y = data["Late_delivery_risk"]

        # --------------------------------------------------
        # 2. Drop leakage & identifier columns
        # --------------------------------------------------
        leakage_columns = [
            "Days for shipping (real)",
            "Days for shipment (scheduled)",
            "Delivery Status",
            "Order Status",
            "shipping date (DateOrders)",
            "Product Status"
        ]

        identifier_columns = [
            "Customer Fname",
            "Customer Lname",
            "Customer Email",
            "Customer Password",
            "Product Image",
            "Customer Zipcode",
            "Customer Id",
            "Order Id",
            "Order Item Id",
            "Order Customer Id",
            "Product Card Id"
        ]

        drop_columns = leakage_columns + identifier_columns
        X = X.drop(columns=[c for c in drop_columns if c in X.columns], errors="ignore")

        # --------------------------------------------------
        # 3. Date feature engineering (ORDER DATE ONLY)
        # --------------------------------------------------
        if "order date (DateOrders)" in X.columns:
            X["order_date"] = pd.to_datetime(X["order date (DateOrders)"])

            X["order_dayofweek"] = X["order_date"].dt.dayofweek
            X["order_month"] = X["order_date"].dt.month
            X["order_is_weekend"] = X["order_dayofweek"].isin([5, 6]).astype(int)

            X = X.drop(columns=["order date (DateOrders)", "order_date"])

        # --------------------------------------------------
        # 4. Drop columns with 100% missing values
        # --------------------------------------------------
        X = X.dropna(axis=1, how="all")

        # --------------------------------------------------
        # 5. Handle missing values
        # --------------------------------------------------
        missing_cols = X.columns[X.isnull().any()].tolist()

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        # Numerical → median
        num_missing_cols = list(set(num_cols).intersection(missing_cols))
        if num_missing_cols:
            X[num_missing_cols] = X[num_missing_cols].fillna(
                X[num_missing_cols].median()
            )

        # Categorical → "Unknown"
        cat_missing_cols = list(set(cat_cols).intersection(missing_cols))
        if cat_missing_cols:
            X[cat_missing_cols] = X[cat_missing_cols].fillna("Unknown")

        # --------------------------------------------------
        # 6. Missing-value indicator features
        # --------------------------------------------------
        for col in missing_cols:
            X[f"{col}_was_missing"] = X[col].isnull().astype(int)

        # --------------------------------------------------
        # 7. Save preprocessed (UNENCODED) data
        # --------------------------------------------------
        X_path = self.output_dir / "X_preprocessed.csv"
        y_path = self.output_dir / "y_preprocessed.csv"

        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False)

        return X, y
