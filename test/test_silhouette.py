# write your silhouette score unit tests here
import cluster
from cluster import kmeans, silhouette, utils
from sklearn.metrics import silhouette_samples
import random
import unittest

# This function tests silhouette score for comparing the true clusters with true labels. 
# This compares the silhouette score calculation with Scikit-learn's function, silhouette_sample which calculates silhouette score for all points. 
# This way, we are testing just the silhouette score without the k-means algorithm. 

def test_silhouette_true():
    decoy = utils.make_clusters(n=1000, k=3)
    s = silhouette.Silhouette()
    true_clusters = decoy[0]
    true_labels = decoy[1]
    scores = s.score(true_clusters, true_labels)
    assert (scores[i] == silhouette_samples(decoy[0], true_labels)[i] for i in range(0, len(scores)))

# This function tests the silhouette score for comparing true clusters with predicted labels. 
# Here, we also test the k-means API. 
    
def test_silhouette_pred():

    decoy = utils.make_clusters(n=1000, k=3)
    true_clusters = decoy[0]
    true_labels = decoy[1]
    kmeans_obj = kmeans.KMeans(k=3)
    print(kmeans_obj.max_iter)

    kmeans_obj.fit(decoy[0])
    predicted_labels = kmeans_obj.predict(decoy[0])
    true_clusters = decoy[0]
    true_labels = decoy[1]
    s = silhouette.Silhouette()

    scores = s.score(true_clusters, predicted_labels)
    assert (scores[i] == silhouette_samples(decoy[0], predicted_labels)[i] for i in range(0, len(scores)))
