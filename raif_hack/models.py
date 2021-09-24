import faiss
import numpy as np


class FaissKNearestNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.k = k
        self.y = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.y = y
        self.index = faiss.IndexFlatL2(x.shape[1])
        self.index.add(x.astype(np.float32))

    def knn(self, x: np.ndarray):
        distances, indices = self.index.search(x.astype(np.float32), k=self.k)
        return distances, indices

    def predict(self, x: np.ndarray):
        distances, neighbors = self.knn(x)
        return self._predict(neighbors, self.y)

    @staticmethod
    def _predict(neighbors, targets):
        predictions = []
        for k_neighbors in neighbors:
            predictions.append(targets[k_neighbors].mean().item())

        return np.array(predictions)
