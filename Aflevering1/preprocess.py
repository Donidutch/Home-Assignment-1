from __future__ import annotations

from typing import Any, Literal
import numpy.typing as npt
from numpy.typing import NDArray
import numpy as np


def create_valsets(
    X: npt.NDArray[Any], y: npt.NDArray[Any], n: int, i: int
) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]]]:
    """_summary_

    Args:
        X (NDArray[Any]): _description_
        y (NDArray[Any]): _description_
        n (int): _description_
        i (int): _description_

    Returns:
        tuple[list, list]: _description_
    """
    valX: list[npt.NDArray[Any]] = []
    valY: list[npt.NDArray[Any]] = []

    for j in range(1, i + 1):
        xx: npt.NDArray[Any] = np.array(X[:, 100 + j * n + 1 : 100 + (j + 1) * n + 1])
        yy: npt.NDArray[Any] = np.array(y[100 + j * n + 1 : 100 + (j + 1) * n + 1])
        valX.append(xx)
        valY.append(yy)
    return valX, valY


def svals(
    n: list[int], X: NDArray[Any], y: NDArray[Any]
) -> tuple[
    list[list[np.ndarray[Any, np.dtype[Any]]]],
    list[list[np.ndarray[Any, np.dtype[Any]]]],
]:
    """_summary_

    Args:
        n (list): _description_
        X (NDArray[Any]): _description_
        y (NDArray[Any]): _description_

    Returns:
        tuple[list, list]: _description_
    """
    svals_x = []
    svals_y = []
    for k in n:
        svals_x.append(create_valsets(X, y, k, 5)[0])
        svals_y.append(create_valsets(X, y, k, 5)[1])
    return svals_x, svals_y


def svals2(
    n: int, X: NDArray[Any], y: NDArray[Any]
) -> tuple[list[NDArray[Any]], list[NDArray[Any]]]:
    """_summary_

    Args:
        n (int): _description_
        X (NDArray[Any]): _description_
        y (NDArray[Any]): _description_

    Returns:
        tuple[list, list]: _description_
    """
    svals_x: list[NDArray[Any]] = []
    svals_y: list[NDArray[Any]] = []

    svals_x.append(create_valsets(X, y, n, 5)[0])
    svals_y.append(create_valsets(X, y, n, 5)[1])
    return svals_x, svals_y


def knn_preprocess(
    X: NDArray[Any], y: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Processing the training data.
    Args:
        X (NDArray[Any]): Training set.
        y (NDArray[Any]): class labels.

    Returns:
        _type_: _description_
    """
    X_train: NDArray[Any] = X[:, 0:100]
    y_train: NDArray[Any] = y[:100]

    return X_train, y_train


def knn_data(
    Xs: NDArray[Any],
    ys: NDArray[Any],
    lx: NDArray[Any],
    mx: NDArray[Any],
    hx: NDArray[Any],
    n: int,
    N: list[int],
) -> tuple[
    NDArray[Any],
    NDArray[Any],
    list[NDArray[Any]],
    list[NDArray[Any]],
    list[NDArray[Any]],
    list[NDArray[Any]],
    list[NDArray[Any]],
    list[NDArray[Any]],
]:
    """_summary_

    Args:
        X (NDArray[Any]): _description_
        y (NDArray[Any]): _description_
        lx (NDArray[Any]): _description_
        mx (NDArray[Any]): _description_
        hx (NDArray[Any]): _description_
        n (int): _description_
        N (list[int]): _description_

    Returns:
        _type_: _description_
    """
    nrows: Literal[784] = 784
    ncols: Literal[1877] = 1877

    X: NDArray[Any] = np.reshape(Xs, (ncols, nrows)).T
    X_light: NDArray[Any] = np.reshape(lx, (ncols, nrows)).T
    X_mod: NDArray[Any] = np.reshape(mx, (ncols, nrows)).T
    X_heavy: NDArray[Any] = np.reshape(hx, (ncols, nrows)).T

    # Processing traning data.
    X_train, y_train = knn_preprocess(X, ys)
    XL_train, yL_train = knn_preprocess(X_light, ys)
    XM_train, yM_train = knn_preprocess(X_mod, ys)
    XH_train, yH_train = knn_preprocess(X_heavy, ys)

    # Creating validation data from corrupted training data.
    svals_x, svals_y = svals(N, X, ys)
    X_val, y_val = svals2(n, X, ys)
    XL_val, yL_val = svals2(n, X_light, ys)
    XM_val, yM_val = svals2(n, X_mod, ys)
    XH_val, yH_val = svals2(n, X_heavy, ys)

    # Training data
    Xtrain_values: list[NDArray[Any]] = [X_train, XL_train, XM_train, XH_train]
    ytrain_values: list[NDArray[Any]] = [y_train, yL_train, yM_train, yH_train]

    # Validation data
    Xtest_values: list[NDArray[Any]] = [*X_val, *XL_val, *XM_val, *XH_val]
    ytest_values: list[NDArray[Any]] = [*y_val, *yL_val, *yM_val, *yH_val]
    return (
        X_train,
        y_train,
        svals_x,
        svals_y,
        Xtrain_values,
        ytrain_values,
        Xtest_values,
        ytest_values,
    )
