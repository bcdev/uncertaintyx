#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
import numpy as np


def to_cor(d: int, U: np.ndarray) -> np.ndarray:  # noqa: N806
    """
    Converts an uncertainty tensor into a correlation tensor.

    :param d: The number of inner tensor dimensions.
    :param U: The uncertainty tensor.
    :returns: The correlation tensor.
    """

    def inner(U: np.ndarray) -> np.ndarray:  # noqa: N806
        """The inner conversion."""
        U_ = U.reshape(size, size)  # noqa: N806
        d_ = np.sqrt(np.diag(U_))
        R_ = U_ / d_[:, np.newaxis]  # noqa: N806
        R_ = R_ / d_[np.newaxis, :]  # noqa: N806
        return R_.reshape(U.shape)

    size = np.prod(U.shape[-d:])
    return np.stack([inner(U_) for U_ in U])


def to_cov(R: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: N806
    """
    Converts a correlation tensor and standard uncertainty
    into a covariance tensor.

    :param R: The correlation tensor.
    :param u: The standard uncertainty.
    :returns: The uncertainty tensor.
    """

    def inner(R: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: N806
        """The inner conversion."""
        R_ = R.reshape(u.size, u.size)  # noqa: N806
        u_ = u.reshape(u.size)
        U_ = u_[np.newaxis, :] * R_ * u_[:, np.newaxis]  # noqa: N806
        return U_.reshape(R.shape)

    return np.stack([inner(R_, u_) for R_, u_ in zip(R, u)])
