#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
"""
Errors-in-variables implementation based on orthogonal
distance regression (ODR). Refer to:

Boggs et al. (1992). User's Reference Guide for ODRPACK
Version 2.01. Software for Weighted Orthogonal Distance
Regression. https://doi.org/10.6028/NIST.IR.4834.

Boggs et al. (1989). Algorithm 676: ODRPACK: software for
weighted orthogonal distance regression. ACM Trans. Math.
Softw. 15, 348–364. https://doi.org/10.1145/76909.76913.
"""

import numpy as np
import odrpack

from ...interface.core import Fitting
from ...interface.core import M
from ...interface.core import Result


class EIV(Fitting):
    """
    Errors-in-variables implementation based on orthogonal
    distance regression (ODR).

    This implementation is intended for problems with up to
    :math:`10^4` to :math:`10^5` data points. Refer to:
    """

    def fit(
        self,
        f: M,
        x: np.ndarray,
        y: np.ndarray,
        *,
        ux: np.ndarray | None = None,
        uy: np.ndarray | None = None,
        max_iter: int = 100,
        **kwargs,
    ) -> Result:
        r"""
        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Standard uncertainties :math:`u(X)`.
        :param uy: Standard uncertainties :math:`u(Y)`.
        :param max_iter: The maximum number of iterations conducted.
        :returns: The fit result.
        """

        def t(g: np.ndarray) -> np.ndarray:
            """Transpose (permute) axes for external API compliance."""
            return np.moveaxis(g, 0, -1) if g.ndim > 1 else g

        def r(
            _: np.ndarray, shape: tuple = (), copy: bool = False
        ) -> np.ndarray:
            """Ravel (reshape) for external API compliance."""
            return (
                np.reshape(_, shape=(-1,) + shape, copy=copy)
                if _.ndim - 1 > len(shape)
                else _
            )

        def u(_: np.ndarray, shape: tuple, copy: bool = False) -> np.ndarray:
            """Unravel (revert) to original shape."""
            return (
                np.reshape(_, shape=shape, copy=copy)
                if _.ndim < len(shape)
                else _
            )

        def w(u: np.ndarray) -> np.ndarray:
            """
            Convert sample uncertainties to inverse variance weights
            for external API compliance.
            """
            return 1.0 / np.square(u)

        def eval(x: np.ndarray, p: np.ndarray) -> np.ndarray:
            """Wrap for external API compliance."""
            return r(f.eval(u(p, k_u), u(x.T, m_u)), n_r).T

        def jac_p(x: np.ndarray, p: np.ndarray) -> np.ndarray:
            """Wrap for external API compliance."""
            return t(r(f.jac_p(u(p, k_u), u(x.T, m_u)), n_r + k_r))

        def jac_x(x: np.ndarray, p: np.ndarray) -> np.ndarray:
            """Wrap for external API compliance."""
            return t(r(f.jac_x(u(p, k_u), u(x.T, m_u)), n_r + m_r))

        p = f.estimate(x, y)

        k_u = p.shape
        m_u = x.shape
        n_u = y.shape
        k_r = (np.prod(k_u[0:]),) if len(k_u) > 0 else ()
        m_r = (np.prod(m_u[1:]),) if len(m_u) > 1 else ()
        n_r = (np.prod(n_u[1:]),) if len(n_u) > 1 else ()

        res = odrpack.odr_fit(
            f=eval,
            xdata=r(x, m_r).T,
            ydata=r(y, n_r).T,
            beta0=r(p),
            weight_x=r(w(ux), m_r).T if ux is not None else None,
            weight_y=r(w(uy), n_r).T if uy is not None else None,
            jac_beta=jac_p,
            jac_x=jac_x,
            maxit=max_iter,
            **kwargs,
        )

        popt = u(res.beta, k_u)
        punc = u(res.sd_beta, k_u)
        pcov = u(res.cov_beta * res.res_var, k_u + k_u)
        rvar = np.var(f.eval(popt, x) - y, axis=0, ddof=popt.size)
        cost = 0.5 * res.sum_square  # standard convention

        return Result(
            f,
            popt=popt,
            punc=punc,
            pcov=pcov,
            rvar=rvar,
            cost=cost,
            info=0 if res.info < 4 else 1,
        )
