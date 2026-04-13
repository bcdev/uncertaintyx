#  Copyright (c) Brockmann Consult GmbH, 2026.
#  License: MIT
import numpy as np
import odrpack

from ...interface.core import Fitting
from ...interface.core import M
from ...interface.core import Result


class ODR(Fitting):
    """
    Orthogonal distance regression.

    A facade to the ODRPACK Python package. Refer to:

    Boggs et al. (1992). User's Reference Guide for ODRPACK Version 2.01.
    Software for Weighted Orthogonal Distance Regression.
    https://doi.org/10.6028/NIST.IR.4834.

    Boggs et al. (1989). Algorithm 676: ODRPACK: software for weighted
    orthogonal distance regression. ACM Trans. Math. Softw. 15, 348–364.
    https://doi.org/10.1145/76909.76913
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

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Uncertainties :math:`u(X) \in \mathbb{R}^{M \times m}`.
        :param uy: Uncertainties :math:`u(Y) \in \mathbb{R}^{M \times n}`.
        :param max_iter: The maximum number of iterations conducted.
        :returns: The fit result.
        """

        def t(g: np.ndarray) -> np.ndarray:
            """Transpose for ODRPACK compliance."""
            return np.moveaxis(g, -1, 0) if g.ndim > 1 else g

        def r(
            _: np.ndarray, shape: tuple = (), copy: bool = False
        ) -> np.ndarray:
            """Ravel for ODRPACK compliance."""
            return (
                np.reshape(_, shape=(-1,) + shape, copy=copy)
                if _.ndim - 1 > len(shape)
                else _
            )

        def u(_: np.ndarray, shape: tuple, copy: bool = False) -> np.ndarray:
            """Unravel for ODRPACK compliance."""
            return (
                np.reshape(_, shape=shape, copy=copy)
                if _.ndim < len(shape)
                else _
            )

        def eval(x: np.ndarray, p: np.ndarray) -> np.ndarray:
            """Wrap for ODRPACK compliance."""
            return r(f.eval(u(p, k_u), u(x.T, m_u)), n_r).T

        def jac_p(x: np.ndarray, p: np.ndarray) -> np.ndarray:
            """Wrap for ODRPACK compliance."""
            return t(r(f.jac_p(u(p, k_u), u(x.T, m_u)), n_r + k_r))

        def jac_x(x: np.ndarray, p: np.ndarray) -> np.ndarray:
            """Wrap for ODRPACK compliance."""
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
            xdata=r(x, m_r, copy=True).T,
            ydata=r(y, n_r, copy=True).T,
            beta0=r(p),
            weight_x=r(1.0 / np.square(ux), m_r).T
            if ux is not None
            else None,
            weight_y=r(1.0 / np.square(uy), n_r).T
            if uy is not None
            else None,
            jac_beta=jac_p,
            jac_x=jac_x,
            maxit=max_iter,
            **kwargs,
        )

        popt = u(res.beta, k_u)
        punc = u(res.sd_beta, k_u)
        pcov = u(res.cov_beta * res.res_var, k_u + k_u)
        rvar = np.var(f.eval(popt, x) - y, axis=0, ddof=popt.size)
        cost = res.sum_square

        return Result(
            f,
            popt=popt,
            punc=punc,
            pcov=pcov,
            rvar=rvar,
            cost=cost,
            info=0 if res.info < 4 else 1,
        )
