import numpy as np
from scipy.spatial.distance import cdist
import random
from sklearn.metrics import silhouette_samples

### Edge cases that you need to handle 
# Make sure k is less than n (observations)
# k relating to m? How does k perform with very high m 
# What is the variance threshold you should set? 

class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None

    # using k++ to initialize farthest points from each other as initial centroids
    def _set_initial_centroids(self, mat: np.ndarray):
        centroid = np.array([mat[random.randint(0, len(mat) - 1)]])
        i = 1
        while (i < self.k):
            minimum_distances = np.min(cdist(mat, centroid, 'euclidean'), axis=1)
            next_centroid = mat[np.argmax(minimum_distances)]
            centroid = np.append(centroid, next_centroid).reshape(i + 1, 2)
            i += 1
        self.centroids = centroid
    
    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # self._set_initial_centroids(mat)
        self.centroids = mat[np.random.choice(mat.shape[0], self.k, replace=False)]
        for i in range(0, self.max_iter):
            centroid_distances = cdist(mat, self.centroids, 'euclidean')
            cluster_ids = np.argmin(centroid_distances, axis=1)
            new_centroids = np.array([mat[cluster_ids == i].mean(axis=0) for i in range(self.k)])

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
        
        if (i >= self.max_iter):
            print("K-Means failed to converge in max iterations.")

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        centroid_distances = cdist(mat, self.centroids, 'euclidean')
        self.cluster_ids = np.argmin(centroid_distances, axis=1)
        self.mat = mat

        return self.cluster_ids
    
    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        sums = []
        for i in range(self.k):
            sums.append(np.sum(np.array(cdist(self.mat[self.cluster_ids == i], [self.centroids[i]], "euclidean")**2)))
        assert (len(sums) == self.k)
        self.SSE = np.sum(sums)
        return self.SSE
            

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids