import numpy as np 
from sklearn.impute import KNNImputer as SklearnKNNImputer

class CustomKNNImputer(SklearnKNNImputer):
    """Customized class for adapt the distance metric for Heterogeneous 
    Eucliedean-Overlap Metric (HEOM) to realize the imputation task
    with KNNImpute from sklearn"""

    def __init__(self, original, featuresType, **kwargs):
        super().__init__(**kwargs)
        self.original = original 
        self.featuresType = featuresType

    def heom_scikit(self,x1:np.array,x2:np.array,missing_values=np.nan):
        "Function to calculate the HEOM: Heterogeneous Euclidean-Overlap Metric for two arrays"
        X = self.original.copy()
        max_xj = X.max()
        min_xj = X.min()
        den = np.array(max_xj) - np.array(min_xj)

        d = [] 
        for k in range(len(self.featuresType)):
            val_a = x1[k]
            val_b = x2[k]
            
            # If values of xA or xB are unknown, d = 1
            if np.isnan(val_a) or np.isnan(val_b):
                d.append(1.0)
            # If all values in both are the same, d = 0
            elif val_a == val_b:
                d.append(0.0)
            # If value is nominal
            elif self.featuresType[k]:
                d.append(1.0)        
            else:
                # If max-min is 0
                if  den[k] == 0:
                    d.append(1.0)
                else:
                    num = np.abs(val_a - val_b)
                    d_cont = num/den[k]
                    d.append(d_cont)

        dj = np.sum(np.array(d) ** 2)
        return np.sqrt(dj)

    def _fit(self, X):
        return super()._fit(X)

    def _transform(self, X):
        return super()._transform(X)

    def fit_transform(self, X, y=None):
        return super().fit_transform(X, y)