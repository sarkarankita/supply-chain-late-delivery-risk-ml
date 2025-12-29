import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders.target_encoder import TargetEncoder
from scipy.sparse import hstack, csr_matrix


class EncodingPipeline:
    def __init__(self, cardinality_threshold=10, scale_numeric=True):
        self.cardinality_threshold = cardinality_threshold
        self.scale_numeric = scale_numeric

        self.num_cols = []
        self.low_card_cols = []
        self.high_card_cols = []

        self.scaler = StandardScaler() if scale_numeric else None
        self.ohe = None
        self.target_encoder = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        self.low_card_cols = []
        self.high_card_cols = []

        for col in cat_cols:
            if X[col].nunique() <= self.cardinality_threshold:
                self.low_card_cols.append(col)
            else:
                self.high_card_cols.append(col)

        if self.num_cols and self.scale_numeric:
            self.scaler.fit(X[self.num_cols])

        if self.low_card_cols:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            self.ohe.fit(X[self.low_card_cols])

        if self.high_card_cols:
            self.target_encoder = TargetEncoder(smoothing=20)
            self.target_encoder.fit(X[self.high_card_cols], y)

        return self

    def transform(self, X: pd.DataFrame):
        if self.ohe is None and self.target_encoder is None and not self.num_cols:
            raise RuntimeError("EncodingPipeline must be fitted before transform.")

        parts = []

        if self.num_cols:
            X_num = X[self.num_cols]
            if self.scale_numeric:
                X_num = self.scaler.transform(X_num)
            parts.append(csr_matrix(X_num))

        if self.low_card_cols and self.ohe:
            parts.append(self.ohe.transform(X[self.low_card_cols]))

        if self.high_card_cols and self.target_encoder:
            X_te = self.target_encoder.transform(X[self.high_card_cols]).values
            parts.append(csr_matrix(X_te))

        return hstack(parts).tocsr()

    def get_feature_names(self):
        names = list(self.num_cols)

        if self.low_card_cols and self.ohe:
            names.extend(self.ohe.get_feature_names_out(self.low_card_cols))

        if self.high_card_cols:
            names.extend([f"{c}_te" for c in self.high_card_cols])

        return names
