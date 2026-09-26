import joblib
import os
from datetime import date
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataTransformator():
    def __init__(self, log_transform_cols: list[str], scale_cols: list[str], one_hot_cols: list[str], label_cols: list[str]) -> None:
        self.log_transform_cols = log_transform_cols
        self.scale_cols = scale_cols
        self.one_hot_cols = one_hot_cols
        self.label_cols = label_cols
        self._build_transformers()

    def _build_transformers(self):
        log_scale_pipeline = Pipeline(steps=[
            ("log_transform", FunctionTransformer(np.log1p, validate=True)),
            ("scale", RobustScaler())
        ])

        self.continuous_transformer = ColumnTransformer(
            transformers=[
                ("log_scaled", log_scale_pipeline, self.log_transform_cols),
                ("standard_scaled", RobustScaler(), self.scale_cols)
            ],
            remainder="drop"
        )

        self.discrete_transformer = ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.one_hot_cols),
                ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.label_cols)
            ],
            remainder="drop"
        )

    def fit_transform(self, df_train: pd.DataFrame, save_path='outputs/models/transformator.pt'):
        continuous_features = self.continuous_transformer.fit_transform(df_train).astype(np.float32)
        discrete_features = self.discrete_transformer.fit_transform(df_train).astype(np.float32)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self, save_path)

        return continuous_features, discrete_features

    def transform(self, df: pd.DataFrame):
        continuous_features = self.continuous_transformer.transform(df).astype(np.float32)
        discrete_features = self.discrete_transformer.transform(df).astype(np.float32)

        return continuous_features, discrete_features
