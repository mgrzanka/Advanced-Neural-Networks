import joblib
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class DataTransformer:
    def __init__(self, to_scale_cols, to_encode_cols, other_cols) -> None:
        self.transformer = self._build_pipeline(to_scale_cols, to_encode_cols, other_cols)

    def _build_pipeline(self, to_scale_cols, to_encode_cols, other_cols):
        scaling_stream = Pipeline(steps=[
            ('fill-missing-values', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ])
        encoding_stream = Pipeline(steps=[
            ('fill-missing-values', SimpleImputer(strategy='median')),
            ('one-hot-encode', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformer = ColumnTransformer(transformers=[
            ('scaling-transformer', scaling_stream, to_scale_cols),
            ('encoding-transformer', encoding_stream, to_encode_cols),
            ('passthrough-others', 'passthrough', other_cols)
        ], remainder='drop')

        return transformer

    def fit_transform(self, train_df, save_path='outputs/models/transformer.pt'):
        transformed = self.transformer.fit_transform(train_df)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        joblib.dump(self, filename=save_path)

        return transformed

    def transform(self, df):
        return self.transformer.transform(df)
