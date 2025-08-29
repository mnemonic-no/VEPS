import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import ast


class StringToListTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda series: series.apply(self.parse_string_to_list))

    @staticmethod
    def parse_string_to_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []

    def get_feature_names_out(self, input_features=None):
        return input_features


class CWEBinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.all_cwes = None

    def fit(self, X, y=None):
        all_cwes = set()
        for cwes in X.iloc[:, 0]:
            if isinstance(cwes, list):
                all_cwes.update(
                    cwe
                    for cwe in cwes
                    if isinstance(cwe, str) and cwe.startswith("CWE-")
                )
        self.all_cwes = sorted(all_cwes)
        return self

    def transform(self, X):
        cwe_data = {
            cwe: X.iloc[:, 0].apply(lambda cwes: 1 if cwe in cwes else 0)
            for cwe in self.all_cwes
        }
        X_encoded = pd.DataFrame(cwe_data, index=X.index)
        return X_encoded.astype(pd.SparseDtype("int", 0))

    def get_feature_names_out(self, input_features=None):
        return np.array(self.all_cwes)


def safe_log(X: np.ndarray) -> np.ndarray:
    """Apply safe logarithm transformation to input data."""
    return np.log1p(X)
