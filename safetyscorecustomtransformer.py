from sklearn.base import BaseEstimator, TransformerMixin

class SafetyScoreAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_safety_score=True):
        self.add_safety_score = add_safety_score

    def fit(self, X, y=None):
        return self  #nothing else to do

    def transform(self, X):
        if self.add_safety_score:
            # Assigning custom weights to features based on relevance to safety. 5 being the highest weight,
            # 3 has an impact to some degree and 1 somewhat being less relevant
            safety_score = (1 / (X["Traffic_Density"] + 1)) * 1 + \
                           (1 / (X["Speed_Limit"] + 1)) * 3 + \
                           (X["Accident_Severity"]) * 5
            X["Safety_Score"] = safety_score
        return X