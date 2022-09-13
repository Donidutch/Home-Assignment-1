from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any
from typing_extensions import reveal_type

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def h_prime(X: npt.NDArray[Any], a: float, b: float) -> npt.NDArray[Any]:
    """An affine model.

    Args:
        X (npt.NDArray[Any]): Training data.
        a (float): coeffs.
        b (float): bias.

    Returns:
        npt.NDArray[Any]: The affine model.
    """
    return a * X + b


def h_nonlin(X: npt.NDArray[Any], a: float, b: float) -> npt.NDArray[Any]:
    """Builds a non-linear function, where the non-linear input is transformed by the mapping x -> sqrt(x)

    Args:
        X (npt.NDArray[Any]): Training data.
        a (float): coeffs.
        b (float): bias.

    Returns:
        npt.NDArray[Any]: The non-linear model.
    """
    return np.exp(np.sqrt(X) * a + b)


def h(X: npt.NDArray[Any], a: float, b: float) -> npt.NDArray[Any]:
    """Constructing non-linear model by using the parameters a and b.

    Args:
        X (npt.NDArray[Any]): Traning data
        a (float): _description_
        b (float): the bias

    Returns:
        npt.NDArray[Any]: Returns the non-linear model.
    """
    return np.exp(a * X + b)


def MSE(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    f: Callable[[npt.NDArray[Any], float, float], npt.NDArray[Any]],
    a: float,
    b: float,
) -> np.floating[Any]:
    """Computing the mean-squared-error of the model f over the training data.

    Args:
        X (npt.NDArray[Any]): Training data.
        y (npt.NDArray[Any]): Training labels.
        f (function): A model.
        a (float): _description_
        b (float): bias

    Returns:
        float: The MSE.
    """
    e = y - f(X, a, b)
    error: np.floating[Any] = np.mean(np.square(e))
    return error


def R2_score(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    f: Callable[[npt.NDArray[Any], float, float], npt.NDArray[Any]],
    a: float,
    b: float,
) -> np.floating[Any]:
    """Computes the R2 score...

    Args:
        X (npt.NDArray[Any]): Training data.
        y (npt.NDArray[Any]): Training labels.
        f (Callable): A model.
        a (float): _description_
        b (float): bias

    Returns:
        float: The R2 score.
    """
    hx: npt.NDArray[Any] = f(X, a, b)
    num = ((y - hx) ** 2).sum(0)
    denom = ((y - np.mean(y)) ** 2).sum(0)
    ret = np.float64(1.0 - (num / denom))

    return ret


def plot_reg(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    f: Callable[[npt.NDArray[Any], float, float], npt.NDArray[Any]],
    a: float,
    b: float,
    yscale_log: bool,
) -> None:
    """_summary_

    Args:
        X (npt.NDArray[Any]): _description_
        y (npt.NDArray[Any]): _description_
        h (Callable): _description_
        a (float): _description_
        b (float): _description_
        yscale_log (bool): _description_
    """

    plt.scatter(X, y)
    plt.plot(np.unique(X), f(np.unique(X), a, b), "r")
    plt.xlabel("Age")
    plt.ylabel("PCB concentration")
    if yscale_log:
        plt.yscale("log")


def plot_digits(imgs: list[npt.NDArray[Any]]) -> None:
    """
    Plot images of MNIST numbers.

    Args:
        imgs (list): list of MNIST image
    """
    fig, ax = plt.subplots(1, 4)
    for i, ax in enumerate(ax.flatten()):
        img = imgs[i][:, 0]
        plottable_image = np.reshape(img, (28, 28))
        ax.imshow(plottable_image.T, cmap="gray_r")


def plot_var(K: npt.ArrayLike, var: list[Any], i: int, b) -> None:
    figname = "var"
    if b:
        figname = "VarAllset"
    n: list[int] = [10, 20, 40, 80]
    for j in range(i):

        plt.plot(K, var[j], label="n = %i" % (n[j]))
        plt.legend()
        plt.xlabel("K")
        plt.ylabel("Variance of validation error")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(figname)


def plot_error(
    K: npt.ArrayLike, errors: list[list[list[float]]], n: int | list, m: int, b: bool
) -> None:
    fig, axx = plt.subplots(2, 2, figsize=(15, 12))

    N = [10, 20, 40, 80]
    ax = np.ravel(axx)
    title = "Error for "
    figname = "Task1"
    if b:
        titles = [
            "Uncorrupted set",
            "Lightly corrupted set",
            "Moderately corrupted set",
            "Heavily corrupted set",
        ]
        figname = "Task2"
    else:
        title = "Uncorrupted set "
    for i in range(n):
        Nn = N[i]
        if b:
            title = titles[i]
            Nn = 80
        for j in range(m):
            ax[i].plot(K, errors[i][j], label="$S_%i$" % (j + 1))
            ax[i].legend()
            ax[i].set_xlabel("K")
            ax[i].set_ylabel("Validation error")
            ax[i].grid(True)
        ax[i].set_title("Error for " + title + " for n = %i" % (Nn))
    plt.tight_layout()
    plt.savefig(figname)
