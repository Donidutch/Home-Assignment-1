from __future__ import annotations


import numpy as np
import numpy.typing as npt
from typing import Any
from numpy import ndarray
from numpy.typing import NDArray


class LinearRegression:
    """
    Linear regression implementation.
    """

    def __init__(self) -> None:

        pass

    def fit(self, Xs: npt.NDArray[Any], ys: npt.NDArray[Any]) -> None:
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, 1]
        y : Array of shape [n_samples, 1]
        """
        x: npt.NDArray[Any] = np.asanyarray(Xs).reshape((len(Xs), -1))
        # X: ndarray[Any, np.dtype[Any]] = np.array(X).reshape((len(X), -1))
        ones: npt.NDArray[np.float64] = np.ones((x.shape[0], 1))
        X: NDArray[Any] = np.concatenate((ones, x), axis=1)
        y: ndarray[Any, np.dtype[Any]] = np.array(ys).reshape((len(ys), 1))
        self.w: NDArray[np.float_] = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, Xs: npt.NDArray[Any]) -> NDArray[np.floating[Any]]:
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        Xs : Array of shape [n_samples, 1]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """

        X: ndarray[Any, np.dtype[Any]] = np.array(Xs).reshape(len(Xs), -1)
        ones: NDArray[np.float64] = np.ones((X.shape[0], 1))
        X_new: NDArray[Any] = np.concatenate((ones, X), axis=1)
        predictions: NDArray[np.floating[Any]] = self.w @ X_new
        return predictions
