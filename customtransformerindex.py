import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RiskIndexAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_risk_index=True):
        self.add_risk_index = add_risk_index

    def fit(self, X, y=None):
        return self  #nothing else to do

    def transform(self, X):
        return X
