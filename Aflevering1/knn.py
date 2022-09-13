import collections

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from typing import Any


class KNN:
    """KNN classifier."""

    def __init__(self, K: int) -> None:
        self.K: int = K

    def fit(
        self, X_train: npt.NDArray[np.float_], y_train: npt.NDArray[np.float_]
    ) -> None:
        """Fitting the KNN from training data.

        Args:
            X_train (npt.NDArray[np.float_]): Training data.
            y_train (npt.NDArray[np.float_]): Labels.
        """
        self.X_train: NDArray[np.float_] = X_train
        self.y_train: NDArray[np.float_] = y_train

    def predict(self, X_test: npt.NDArray[np.float_]) -> float:
        """
        Computes prediction of class labels for the dataset

        Args:
            X_test (npt.NDArray[np.float_]): The test point
            k (int): The number of neighbors

        Returns:
            float: The prediction.
        """
        distances: list[np.float_] = []
        neighbors: list[Any] = []

        n: int = len(self.X_train.T)
        k: int = self.K

        for i in range(n):

            dis: np.floating[Any] = np.linalg.norm(X_test - self.X_train[:, i])

            distances.append(dis)

        distance: NDArray[np.float_] = np.array(distances)
        kneighbor: np.ndarray[Any, Any] = distance.argsort()[:k]

        for j in kneighbor:
            neighbors.append(self.y_train[j])
        pred: float = collections.Counter(neighbors).most_common(1)[0][0]

        # pred = stats.mode(neighbors,keepdims=True)[0][0]
        return pred

    np.float64

    def accuracy(
        self, y_test: npt.NDArray[np.float_], predictions: list[int | float]
    ) -> float:
        """Computes the accuracy.

        Args:
            y_test (npt.NDArray[np.float_]): Test labels.
            predictions (list): The predictions made by the KNN classifier.

        Returns:
            float: The accuracy.
        """

        n: int = len(y_test)
        correct: float = 0.0
        for x in range(n):
            if y_test[x] == predictions[x]:
                correct += 1.0
        return correct / float(n)

    def error(
        self, y_test: npt.NDArray[np.float_], predictions: list[float | int]
    ) -> float:
        """Computes the validation error.

        Args:
            y_test (npt.NDArray[np.float_]): Test labels
            predictions (list):The predictions made by the KNN classifier.

        Returns:
            float: The validation error.
        """
        n: int = len(y_test)

        correct: float = 0.0
        for x in range(n):
            if y_test[x] != predictions[x]:
                correct += 1.0
        return correct / float(n)
