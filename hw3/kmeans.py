
import math
import random
from typing import List

class KMeansClusterClassifier:
    def __init__(self, n_cluster: int):
        self.n_clusters = n_cluster
        self.centroids = []
        self.labels = []

    def fit(self, X: List[List[float]], y: List[int]):
        # Initialize centroids randomly
        self.centroids = random.sample(X, self.n_clusters)

        for _ in range(100):  # Maximum number of iterations
            clusters = [[] for _ in range(self.n_clusters)]
            cluster_labels = [[] for _ in range(self.n_clusters)]

            # Assign points to the nearest centroid
            for point, label in zip(X, y):
                distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
                closest_centroid_index = distances.index(min(distances))
                clusters[closest_centroid_index].append(point)
                cluster_labels[closest_centroid_index].append(label)

            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:  # Avoid division by zero
                    new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
                    new_centroids.append(new_centroid)
                else:  # If a cluster is empty, keep the old centroid
                    new_centroids.append(self.centroids[clusters.index(cluster)])

            # Check for convergence
            if self.centroids == new_centroids:
                break
            self.centroids = new_centroids

        # Determine the label for each centroid based on majority vote
        self.labels = []
        for labels in cluster_labels:
            if labels:
                self.labels.append(max(set(labels), key=labels.count))
            else:
                self.labels.append(None)

    def predict(self, X: List[List[float]]):
        predictions = []
        for point in X:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid_index = distances.index(min(distances))
            predictions.append(self.labels[closest_centroid_index])
        return predictions

    def assign_clusters(self, X: List[List[float]]):
        clusters = [[] for _ in range(self.n_clusters)]
        for point in X:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid_index = distances.index(min(distances))
            clusters[closest_centroid_index].append(point)
        return clusters

    @staticmethod
    def euclidean_distance(point1: List[float], point2: List[float]) -> float:
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

