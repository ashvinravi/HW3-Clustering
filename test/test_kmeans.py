# Write your k-means unit tests here

from sklearn.model_selection import train_test_split
import cluster
from cluster import kmeans, silhouette, utils
import random
from sklearn.metrics import silhouette_samples
import unittest
import pytest
import numpy as np

# Ensure that K-Means will throw error if k is greater than or equal to number of observations. 
def test_kmeans_high_k():
    with pytest.raises(ValueError):
        k = kmeans.KMeans(k=0)
        k.fit()
def test_kmeans_0():
    with pytest.raises(ValueError):
        k = kmeans.KMeans(k=0)
        decoy = utils.make_clusters(n=10, k=10)
        k.fit(decoy[0])

# If k=1, all of your labels should be in the same cluster. 
def test_kmeans_1():
    decoy = utils.make_clusters(n=1000, k=1)
    s = silhouette.Silhouette()
    training_points, test_points = train_test_split(decoy[0], test_size=0.2, random_state=50)
    training_clusters, test_clusters = train_test_split(decoy[1], test_size=0.2, random_state=50)
    kmeans_obj = kmeans.KMeans(k=1)
    kmeans_obj.fit(training_points)
    predicted_labels = kmeans_obj.predict(test_points)
    assert ( [test_clusters[i] == predicted_labels[i] for i in range(0, len(predicted_labels))] )

# Silhouette score metric: 0.2 = weak, 0.4 = barely acceptable, 0.6 = good, 0.8 = very good
# test k-means for very tightly clustered data (scale=0.1)
def test_kmeans_tight():
    tight_cluster = utils.make_clusters(n=1000, k=3, scale=0.1)
    s = silhouette.Silhouette()
    training_points, test_points = train_test_split(tight_cluster[0], test_size=0.2, random_state=50)
    training_clusters, test_clusters = train_test_split(tight_cluster[1], test_size=0.2, random_state=50)
    kmeans_obj = kmeans.KMeans(k=3)
    kmeans_obj.fit(training_points)
    predicted_labels = kmeans_obj.predict(test_points)
    scores = s.score(test_points, predicted_labels)
    assert (np.mean(scores) >= 0.8)

# test k-means for not well clustered data (scale=2), model should not perform as well (using benchmark of mean score > 0.4)
def test_kmeans_loose():
    loose_cluster = utils.make_clusters(n=1000, k=3, scale=2)
    s = silhouette.Silhouette()
    training_points, test_points = train_test_split(loose_cluster[0], test_size=0.2, random_state=50)
    training_clusters, test_clusters = train_test_split(loose_cluster[1], test_size=0.2, random_state=50)
    kmeans_obj = kmeans.KMeans(k=3)
    kmeans_obj.fit(training_points)
    predicted_labels = kmeans_obj.predict(test_points)
    scores = s.score(test_points, predicted_labels)

    # utils.plot_multipanel(test_points, test_clusters, predicted_labels, scores)
    assert (np.mean(scores) >= 0.4)
def test_kmeans_split():
    decoy = utils.make_clusters(n=100, k=3)
    s = silhouette.Silhouette()
    training_points, test_points = train_test_split(decoy[0], test_size=0.2, random_state=50)
    training_clusters, test_clusters = train_test_split(decoy[1], test_size=0.2, random_state=50)
    kmeans_obj = kmeans.KMeans(k=3)
    kmeans_obj.fit(training_points)
    predicted_labels = kmeans_obj.predict(test_points)
    scores = s.score(test_points, predicted_labels)

    assert (np.mean(scores) >= 0.6)

# test how well kmeans performs with high-dimensional data (200 features) - it should not perform well 
def test_kmeans_hidim():
    hidim_data = utils.make_clusters(n=1000, m=200, k=3)
    s = silhouette.Silhouette()
    training_points, test_points = train_test_split(hidim_data[0], test_size=0.2, random_state=50)
    training_clusters, test_clusters = train_test_split(hidim_data[1], test_size=0.2, random_state=50)
    kmeans_obj = kmeans.KMeans(k=3)
    kmeans_obj.fit(training_points)
    predicted_labels = kmeans_obj.predict(test_points)
    scores = s.score(test_points, predicted_labels)
    # utils.plot_multipanel(test_points, test_clusters, predicted_labels, scores)
    assert (np.mean(scores) >= 0.2)

# Does k-means work with higher number of clusters? 
def test_kmeans_hi_clusters():
    hidim_data = utils.make_clusters(n=1000, k=15)
    s = silhouette.Silhouette()
    training_points, test_points = train_test_split(hidim_data[0], test_size=0.2, random_state=50)
    training_clusters, test_clusters = train_test_split(hidim_data[1], test_size=0.2, random_state=50)
    kmeans_obj = kmeans.KMeans(k=15)
    kmeans_obj.fit(training_points)
    predicted_labels = kmeans_obj.predict(test_points)
    scores = s.score(test_points, predicted_labels)
    # utils.plot_multipanel(test_points, test_clusters, predicted_labels, scores)
    assert (np.mean(scores) >= 0.4)
