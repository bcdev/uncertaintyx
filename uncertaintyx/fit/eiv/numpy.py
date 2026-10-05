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

JCGM 100:2008. Evaluation of measurement data - Guide to the
expression of uncertainty in measurement.
https://doi.org/10.59161/JCGM100-2008E

JCGM 101:2008. Supplement 1 to the Guide to the expression
of uncertainty in measurement - Propagation of distributions
using a Monte Carlo method. https://doi.org/10.59161/JCGM101-2008

JCGM 102:2011. Supplement 2 to the Guide to the expression of
uncertainty in measurement - Extension to any number of output
quantities. https://doi.org/10.59161/JCGM102-2011

JCGM GUM-6:2020. Guide to the expression of uncertainty in
measurement — Part 6: Developing and using measurement models.
https://doi.org/10.59161/JCGMGUM-6-2020
"""

import numpy as np
import odrpack

from ...tyx import Fitted
from ...tyx import Fitting
from ...tyx import M


class EIV(Fitting):
    """
    Implements an errors-in-variables inversion via Orthogonal
    Distance Regression (ODR) compliant with JCGM 100:2008 and
    102:2011. By simultaneously optimizing parameters and latent
    variables, it evaluates the global covariance matrix at the
    solution minimum (GUM-6:2020).
    """

    def fit(
        self,
        f: M,
        x: np.ndarray,
        y: np.ndarray,
        *,
        ux: np.ndarray | None = None,
        uy: np.ndarray | None = None,
        max_steps: int = 100,
        **kwargs,
    ) -> Fitted:
        r"""
        This function does not belong to public API.

        Fits the parameters of a model function to :math:`M`
        samples :math:`(x_i, y_i)` of data.

        Under the same notation and remarks as :class:`M`:

        :param f: The model function.
        :param x: Samples :math:`X \in \mathbb{R}^{M \times m}`.
        :param y: Samples :math:`Y \in \mathbb{R}^{M \times n}`.
        :param ux: Standard uncertainties :math:`u(X)`.
        :param uy: Standard uncertainties :math:`u(Y)`.
        :param max_steps: The maximum number of steps the optimizer can take.
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

        p = f.prior(x, y)

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
            maxit=max_steps,
            **kwargs,
        )

        popt = u(res.beta, k_u)
        """
        Expectation of the posterior PDF (MAP estimator, JCGM 101:2008).
        """
        punc = u(res.sd_beta, k_u)
        """
        Standard uncertainties extracted from diagonal (JCGM 100/102).
        """
        pcov = u(res.cov_beta * res.res_var, k_u + k_u)
        """
        Covariance matrix via Laplace Approximation (JCGM 102:2011).
        """
        zvar = np.var(f.eval(popt, x) - y, axis=0, ddof=popt.size)
        cost = np.asarray(0.5 * res.sum_square)  # standard convention

        return Fitted(
            f,
            popt=popt,
            pcov=pcov,
            punc=punc,
            zvar=zvar,
            cost=cost,
            info=0 if res.info < 4 else 1,
        )
