# src/utils/custom_transformers.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Um transformer customizado para calcular 'opportunity_duration_days'
    a partir de 'engage_date' e 'close_date', e tratar seus NaNs/valores <= 0.
    """
    def __init__(self, median_duration=None):
        self.median_duration = median_duration # Para armazenar a mediana aprendida no fit

    def fit(self, X, y=None):
        # Converte as colunas de data para datetime e calcula a duração.
        temp_df = X.copy()
        temp_df['engage_date_dt'] = pd.to_datetime(temp_df['engage_date'], errors='coerce')
        temp_df['close_date_dt'] = pd.to_datetime(temp_df['close_date'], errors='coerce')
        temp_df['duration'] = (temp_df['close_date_dt'] - temp_df['engage_date_dt']).dt.days

        # Calcula a mediana de durações válidas e positivas para imputação.
        valid_durations = temp_df['duration'][temp_df['duration'] > 0]
        if not valid_durations.empty:
            self.median_duration = valid_durations.median()
        else:
            self.median_duration = 0 # Fallback seguro

        return self

    def transform(self, X):
        X_transformed = X.copy()

        X_transformed['engage_date_dt'] = pd.to_datetime(X_transformed['engage_date'], errors='coerce')
        X_transformed['close_date_dt'] = pd.to_datetime(X_transformed['close_date'], errors='coerce')

        X_transformed['opportunity_duration_days'] = (
            X_transformed['close_date_dt'] - X_transformed['engage_date_dt']
        ).dt.days

        imputation_value = self.median_duration if self.median_duration is not None else 0
        
        X_transformed['opportunity_duration_days'] = X_transformed['opportunity_duration_days'].apply(
            lambda x: imputation_value if pd.isna(x) or x <= 0 else x
        )

        columns_to_drop = ['engage_date', 'close_date', 'engage_date_dt', 'close_date_dt']
        X_transformed = X_transformed.drop(columns=[col for col in columns_to_drop if col in X_transformed.columns], errors='ignore')

        return X_transformed