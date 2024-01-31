import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        self.sil_score = None

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        silhouette_scores = []
        for i in range(0, len(X)):
            a = np.array(cdist([X[i]], X[y == y[i]], 'euclidean'))
            a = np.mean(a[a != 0])
            cluster_ids = np.unique(np.array(y))
            cluster_ids = cluster_ids[cluster_ids != y[i]]  
            b = np.min([np.mean(cdist(X[y == cluster],[X[i]], "euclidean")) for cluster in cluster_ids])
            denominator = max(a, b)
            silhouette_scores.append( (b - a) / denominator ) 
        self.silhouette_scores = silhouette_scores
        return  self.silhouette_scores
