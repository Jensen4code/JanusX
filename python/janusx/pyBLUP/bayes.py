from __future__ import annotations
from typing import Optional, Tuple
import typing
import warnings
import numpy as np

from janusx.janusx import bayesa as _bayesa, bayesb as _bayesb, bayesc as _bayesc
from janusx.pyBLUP.mlm import BLUP

_BAYESA_MIN_ABS_BETA_WARNED = False

# Production GS defaults.  The first value is the hard upper bound for the
# R-hat monitoring chain; the second is the post-convergence burn-in count.
BAYES_MCMC_DEFAULTS: dict[str, tuple[int, int]] = {
    "BayesA": (3000, 1000),
    "BayesB": (3000, 1000),
    "BayesC": (3000, 1000),
}


def bayes_mcmc_defaults(method: str) -> tuple[int, int]:
    """Return ``(rhat_max_iter, post_convergence_burnin)`` defaults."""
    key = str(method).strip()
    try:
        return BAYES_MCMC_DEFAULTS[key]
    except KeyError as exc:
        supported = ", ".join(BAYES_MCMC_DEFAULTS)
        raise ValueError(f"Unsupported Bayes method {method!r}; use {supported}.") from exc


def _scalar_from_native(value: object, *, integer: bool = False) -> float | int | None:
    """Convert a scalar PyO3 diagnostic without accepting malformed arrays."""
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if int(arr.size) != 1 or not np.isfinite(float(arr[0])):
            return None
        scalar = float(arr[0])
        if integer:
            if scalar < 0.0 or not np.isclose(scalar, round(scalar)):
                return None
            return int(round(scalar))
        return scalar
    except Exception:
        return None


def _parse_bayes_diagnostics(
    method: str,
    diag: typing.Sequence[object],
    beta_size: int,
) -> dict[str, object]:
    """Parse legacy and current native Bayes diagnostic tails.

    Current non-trace kernels append ``rhat, actual_iterations,
    post_burnin_start_iteration`` for BayesA and ``pip, rhat, actual_iterations,
    post_burnin_start_iteration`` for BayesB/BayesC.  The length-aware parser
    keeps older installed extensions usable while avoiding the p==1 scalar PIP
    ambiguity.
    """
    tail = list(diag)
    pip_item: object | None = None
    rhat_item: object | None = None
    actual_item: object | None = None
    post_start_item: object | None = None
    method_key = str(method).strip().lower()
    if method_key in {"bayesb", "bayesc"}:
        if len(tail) >= 6:
            pip_item, rhat_item, actual_item, post_start_item = tail[-4:]
        elif len(tail) >= 4:
            pip_item, rhat_item = tail[-2:]
        elif len(tail) >= 3:
            # Pre-R-hat B/C ABI: prob_in, n_active, pip.
            pip_item = tail[-1]
    elif len(tail) >= 3:
        rhat_item, actual_item, post_start_item = tail[-3:]
    elif tail:
        rhat_item = tail[-1]

    pip: np.ndarray | None = None
    if pip_item is not None:
        try:
            candidate = np.asarray(pip_item, dtype=np.float64).reshape(-1)
            if (
                int(candidate.size) == int(beta_size)
                and np.all(np.isfinite(candidate))
                and np.all((candidate >= 0.0) & (candidate <= 1.0))
            ):
                pip = np.ascontiguousarray(candidate, dtype=np.float64)
        except Exception:
            pip = None
    rhat = _scalar_from_native(rhat_item)
    actual = _scalar_from_native(actual_item, integer=True)
    post_start = _scalar_from_native(post_start_item, integer=True)
    return {
        "pip": pip,
        "rhat_h2": float(rhat) if rhat is not None else float("nan"),
        "actual_iterations": int(actual) if actual is not None else 0,
        "post_burnin_start_iteration": int(post_start) if post_start is not None else 0,
    }


def _as_1d_f64(arr: np.ndarray, name: str) -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 0:
        raise ValueError(f"{name} must be 1D array-like")
    return np.ascontiguousarray(out.reshape(-1))


def _as_2d_f64(
    arr: np.ndarray,
    name: str,
    n_rows: int,
    *,
    allow_1d: bool = False,
) -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    out = np.asarray(arr, dtype=np.float64)
    if allow_1d and out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if out.shape[0] != n_rows:
        raise ValueError(f"{name} rows must match len(y)")
    return np.ascontiguousarray(out)

def _as_2d_f64_mxn(arr: np.ndarray, name: str, n_cols: int) -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if out.shape[1] != n_cols:
        raise ValueError(f"{name} cols must match len(y)")
    return np.ascontiguousarray(out)


def _validate_fixed_pi(pi: Optional[float]) -> Optional[float]:
    """Validate the optional fixed inclusion probability used by BayesB/C."""
    if pi is None:
        return None
    value = float(pi)
    if not np.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError("pi must be finite and in (0, 1)")
    return value


def _call_bayesa(
    y: np.ndarray,
    m: np.ndarray,
    x: Optional[np.ndarray],
    n_iter: int,
    burnin: int,
    r2: float,
    df0_b: float,
    shape0: float,
    rate0: Optional[float],
    s0_b: Optional[float],
    df0_e: float,
    s0_e: Optional[float],
    min_abs_beta: float,
    seed: Optional[int],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    int,
    int,
]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    thin = 1
    if n_iter <= burnin:
        raise ValueError("n_iter must be > burnin")
    if not np.isfinite(float(min_abs_beta)) or float(min_abs_beta) < 0.0:
        raise ValueError("min_abs_beta is deprecated/ignored; keep it finite and >= 0")
    if not (0.0 < r2 < 1.0):
        raise ValueError("r2 must be in (0, 1)")
    if df0_b <= 0.0 or df0_e <= 0.0:
        raise ValueError("df0_b and df0_e must be > 0")
    if shape0 <= 0.0:
        raise ValueError("shape0 must be > 0")
    if rate0 is not None and rate0 <= 0.0:
        raise ValueError("rate0 must be > 0")
    if s0_b is not None and s0_b <= 0.0:
        raise ValueError("s0_b must be > 0")
    if s0_e is not None and s0_e <= 0.0:
        raise ValueError("s0_e must be > 0")
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")
    global _BAYESA_MIN_ABS_BETA_WARNED
    if not _BAYESA_MIN_ABS_BETA_WARNED:
        warnings.warn(
            "BayesA `min_abs_beta` is deprecated and ignored by Rust backend; it will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        _BAYESA_MIN_ABS_BETA_WARNED = True

    return _bayesa(
        y=y,
        m=m,
        x=x,
        n_iter=n_iter,
        burnin=burnin,
        thin=thin,
        r2=float(r2),
        df0_b=float(df0_b),
        shape0=float(shape0),
        rate0=rate0,
        s0_b=s0_b,
        df0_e=float(df0_e),
        s0_e=s0_e,
        min_abs_beta=float(min_abs_beta),
        seed=seed,
    )


def _call_bayesb(
    y: np.ndarray,
    m: np.ndarray,
    x: Optional[np.ndarray],
    n_iter: int,
    burnin: int,
    r2: float,
    df0_b: float,
    shape0: float,
    rate0: Optional[float],
    s0_b: Optional[float],
    prob_in: float,
    counts: float,
    pi: Optional[float],
    df0_e: float,
    s0_e: Optional[float],
    seed: Optional[int],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    float,
    int,
    int,
]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    thin = 1
    if n_iter <= burnin:
        raise ValueError("n_iter must be > burnin")
    if not (0.0 < r2 < 1.0):
        raise ValueError("r2 must be in (0, 1)")
    if df0_b <= 0.0 or df0_e <= 0.0:
        raise ValueError("df0_b and df0_e must be > 0")
    if shape0 <= 0.0:
        raise ValueError("shape0 must be > 0")
    if rate0 is not None and rate0 <= 0.0:
        raise ValueError("rate0 must be > 0")
    if s0_b is not None and s0_b <= 0.0:
        raise ValueError("s0_b must be > 0")
    if s0_e is not None and s0_e <= 0.0:
        raise ValueError("s0_e must be > 0")
    if not (0.0 < prob_in < 1.0):
        raise ValueError("prob_in must be in (0, 1)")
    if counts < 0.0:
        raise ValueError("counts must be >= 0")
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")

    fixed_pi = _validate_fixed_pi(pi)
    return _bayesb(
        y=y,
        m=m,
        x=x,
        n_iter=n_iter,
        burnin=burnin,
        thin=thin,
        r2=float(r2),
        df0_b=float(df0_b),
        shape0=float(shape0),
        rate0=rate0,
        s0_b=s0_b,
        prob_in=float(prob_in),
        counts=float(counts),
        fixed_pi=fixed_pi,
        df0_e=float(df0_e),
        s0_e=s0_e,
        seed=seed,
    )


def _call_bayesc(
    y: np.ndarray,
    m: np.ndarray,
    x: Optional[np.ndarray],
    n_iter: int,
    burnin: int,
    r2: float,
    df0_b: float,
    s0_b: Optional[float],
    prob_in: float,
    counts: float,
    pi: Optional[float],
    df0_e: float,
    s0_e: Optional[float],
    seed: Optional[int],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    float,
    int,
    int,
]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    thin = 1
    if n_iter <= burnin:
        raise ValueError("n_iter must be > burnin")
    if not (0.0 < r2 < 1.0):
        raise ValueError("r2 must be in (0, 1)")
    if df0_b <= 0.0 or df0_e <= 0.0:
        raise ValueError("df0_b and df0_e must be > 0")
    if s0_b is not None and s0_b <= 0.0:
        raise ValueError("s0_b must be > 0")
    if s0_e is not None and s0_e <= 0.0:
        raise ValueError("s0_e must be > 0")
    if not (0.0 < prob_in < 1.0):
        raise ValueError("prob_in must be in (0, 1)")
    if counts < 0.0:
        raise ValueError("counts must be >= 0")
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")

    fixed_pi = _validate_fixed_pi(pi)
    return _bayesc(
        y=y,
        m=m,
        x=x,
        n_iter=n_iter,
        burnin=burnin,
        thin=thin,
        r2=float(r2),
        df0_b=float(df0_b),
        s0_b=s0_b,
        prob_in=float(prob_in),
        counts=float(counts),
        fixed_pi=fixed_pi,
        df0_e=float(df0_e),
        s0_e=s0_e,
        seed=seed,
    )


def BayesA(
    y: np.ndarray,
    M: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_iter: int = 3000,
    burnin: int = 1000,
    r2: float = 0.5,
    prob_in: float = 0.5,
    counts: float = 5.0,
    df0_b: float = 5.0,
    shape0: float = 1.1,
    rate0: Optional[float] = None,
    s0_b: Optional[float] = None,
    df0_e: float = 5.0,
    s0_e: Optional[float] = None,
    min_abs_beta: float = 1e-9,
    seed: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    int,
    int,
]:
    """
    Python interface for the Rust BayesA kernel (PyO3).

    This wrapper normalizes inputs to contiguous float64 arrays and passes
    them to the Rust implementation `janusx.janusx.bayesa`.

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Phenotype vector. Flattened to 1D float64.
    M : array-like, shape (m, n)
        Marker matrix with markers in rows and samples in columns.
    X : array-like, shape (n, q) or (n,), optional
        Covariate matrix. 1D inputs are treated as a single covariate.
        Include a column of ones here if you want an intercept term. If X is
        None, the Rust backend uses an intercept-only design.
    n_iter : int, default=3000
        Maximum iterations used for R-hat monitoring and posterior sampling.
    burnin : int, default=1000
        Additional burn-in iterations after R-hat reaches the stability threshold.
    r2 : float, default=0.5
        Proportion of variance explained by markers; must be in (0, 1).
    prob_in : float, default=0.5
        Unused for BayesA; kept for API parity.
    counts : float, default=5.0
        Unused for BayesA; kept for API parity.
    df0_b : float, default=5.0
        Prior degrees of freedom for marker effects.
    shape0 : float, default=1.1
        Prior shape parameter for the S update.
    rate0 : float, optional
        Prior rate; if None, computed from data and `shape0`.
    s0_b : float, optional
        Prior scale for marker effects; if None, computed from data.
    df0_e : float, default=5.0
        Prior degrees of freedom for residual variance.
    s0_e : float, optional
        Prior scale for residual variance; if None, derived from data.
    min_abs_beta : float, default=1e-9
        Deprecated and ignored by Rust backend (kept for API compatibility).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Posterior mean marker effects.
    alpha : np.ndarray, shape (q,)
        Posterior mean covariate effects. If `X` includes an intercept column,
        `alpha[0]` corresponds to the intercept.
    varbeta : np.ndarray, shape (p,)
        Posterior mean marker-specific variances.
    varep : float
        Posterior mean residual variance.
    h2_mean : float
        Posterior mean heritability.
    varh2 : float
        Posterior variance of heritability.
    rhat_h2 : float
        Split-chain R-hat of the retained posterior h2 samples.  The Rust
        kernel may enter a post-convergence burn-in stage after repeated values
        below its stability threshold.
    actual_iterations : int
        Number of MCMC updates actually performed, including any post-
        convergence burn-in iterations.
    post_burnin_start_iteration : int
        One-based first iteration of the post-convergence burn-in stage, or 0
        when the R-hat early-stop trigger was not reached.
    Raises
    ------
    ValueError
        If shapes are incompatible or hyperparameters are out of range.

    Notes
    -----
    - Inputs are copied to contiguous float64 arrays before calling Rust.
    - M is expected to be (m, n) with n == len(y).
    """
    y_arr = _as_1d_f64(y, "y")
    m_arr = _as_2d_f64_mxn(M, "M", y_arr.shape[0])
    x_arr = None
    if X is not None:
        x_arr = _as_2d_f64(X, "X", y_arr.shape[0], allow_1d=True)

    return _call_bayesa(
        y_arr,
        m_arr,
        x_arr,
        n_iter,
        burnin,
        r2,
        df0_b,
        shape0,
        rate0,
        s0_b,
        df0_e,
        s0_e,
        min_abs_beta,
        seed,
    )

def BayesB(
    y: np.ndarray,
    M: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_iter: int = 3000,
    burnin: int = 1000,
    r2: float = 0.5,
    prob_in: float = 0.5,
    counts: float = 5.0,
    pi: Optional[float] = None,
    df0_b: float = 5.0,
    shape0: float = 1.1,
    rate0: Optional[float] = None,
    s0_b: Optional[float] = None,
    df0_e: float = 5.0,
    s0_e: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    float,
    int,
    int,
]:
    """
    Python interface for the Rust BayesB kernel (PyO3).

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Phenotype vector. Flattened to 1D float64.
    M : array-like, shape (m, n)
        Marker matrix with markers in rows and samples in columns.
    X : array-like, shape (n, q) or (n,), optional
        Covariate matrix. 1D inputs are treated as a single covariate.
        Include a column of ones here if you want an intercept term. If X is
        None, the Rust backend uses an intercept-only design.
    n_iter : int, default=3000
        Maximum iterations used for R-hat monitoring and posterior sampling.
    burnin : int, default=1000
        Additional burn-in iterations after R-hat reaches the stability threshold.
    r2 : float, default=0.5
        Proportion of variance explained by markers; must be in (0, 1).
    prob_in : float, default=0.5
        Prior inclusion probability for markers.
    counts : float, default=5.0
        Prior strength for inclusion probability.
    pi : float, optional
        Fixed marker inclusion probability. If omitted, the sampler updates
        the inclusion probability from the active-marker counts.
    df0_b : float, default=5.0
        Prior degrees of freedom for marker effects.
    shape0 : float, default=1.1
        Prior shape parameter for the S update.
    rate0 : float, optional
        Prior rate; if None, computed from data and `shape0`.
    s0_b : float, optional
        Prior scale for marker effects; if None, computed from data.
    df0_e : float, default=5.0
        Prior degrees of freedom for residual variance.
    s0_e : float, optional
        Prior scale for residual variance; if None, derived from data.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Posterior mean marker effects.
    alpha : np.ndarray, shape (q,)
        Posterior mean covariate effects. If `X` includes an intercept column,
        `alpha[0]` corresponds to the intercept.
    varbeta : np.ndarray, shape (p,)
        Posterior mean marker-specific variances.
    varep : float
        Posterior mean residual variance.
    h2_mean : float
        Posterior mean heritability.
    varh2 : float
        Posterior variance of heritability.
    prob_in_mean : float
        Posterior mean inclusion probability.
    n_active_mean : float
        Posterior mean number of active markers.
    pip : np.ndarray, shape (p,)
        Posterior inclusion probability for each marker.
    rhat_h2 : float
        Split-chain R-hat of the retained posterior h2 samples.
    actual_iterations : int
        Number of MCMC updates actually performed, including any post-
        convergence burn-in iterations.
    post_burnin_start_iteration : int
        One-based first iteration of the post-convergence burn-in stage, or 0
        when the R-hat early-stop trigger was not reached.
    """
    y_arr = _as_1d_f64(y, "y")
    m_arr = _as_2d_f64_mxn(M, "M", y_arr.shape[0])
    x_arr = None
    if X is not None:
        x_arr = _as_2d_f64(X, "X", y_arr.shape[0], allow_1d=True)

    return _call_bayesb(
        y_arr,
        m_arr,
        x_arr,
        n_iter,
        burnin,
        r2,
        df0_b,
        shape0,
        rate0,
        s0_b,
        prob_in,
        counts,
        pi,
        df0_e,
        s0_e,
        seed,
    )


def BayesC(
    y: np.ndarray,
    M: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_iter: int = 3000,
    burnin: int = 1000,
    r2: float = 0.5,
    prob_in: float = 0.5,
    counts: float = 10.0,
    pi: Optional[float] = None,
    df0_b: float = 5.0,
    s0_b: Optional[float] = None,
    df0_e: float = 5.0,
    s0_e: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    float,
    int,
    int,
]:
    """
    Python interface for the Rust BayesC kernel (PyO3).

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        Phenotype vector. Flattened to 1D float64.
    M : array-like, shape (m, n)
        Marker matrix with markers in rows and samples in columns.
    X : array-like, shape (n, q) or (n,), optional
        Covariate matrix. 1D inputs are treated as a single covariate.
        Include a column of ones here if you want an intercept term. If X is
        None, the Rust backend uses an intercept-only design.
    n_iter : int, default=3000
        Maximum iterations used for R-hat monitoring and posterior sampling.
    burnin : int, default=1000
        Additional burn-in iterations after R-hat reaches the stability threshold.
    r2 : float, default=0.5
        Proportion of variance explained by markers; must be in (0, 1).
    prob_in : float, default=0.5
        Prior inclusion probability for markers.
    counts : float, default=10.0
        Prior strength for inclusion probability.
    pi : float, optional
        Fixed marker inclusion probability. If omitted, the sampler updates
        the inclusion probability from the active-marker counts.
    df0_b : float, default=5.0
        Prior degrees of freedom for marker effects.
    s0_b : float, optional
        Prior scale for marker effects; if None, computed from data.
    df0_e : float, default=5.0
        Prior degrees of freedom for residual variance.
    s0_e : float, optional
        Prior scale for residual variance; if None, derived from data.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Posterior mean marker effects.
    alpha : np.ndarray, shape (q,)
        Posterior mean covariate effects. If `X` includes an intercept column,
        `alpha[0]` corresponds to the intercept.
    varbeta : float
        Posterior mean marker variance (shared across markers).
    varep : float
        Posterior mean residual variance.
    h2_mean : float
        Posterior mean heritability.
    varh2 : float
        Posterior variance of heritability.
    prob_in_mean : float
        Posterior mean inclusion probability.
    n_active_mean : float
        Posterior mean number of active markers.
    pip : np.ndarray, shape (p,)
        Posterior inclusion probability for each marker.
    rhat_h2 : float
        Split-chain R-hat of the retained posterior h2 samples.
    actual_iterations : int
        Number of MCMC updates actually performed, including any post-
        convergence burn-in iterations.
    post_burnin_start_iteration : int
        One-based first iteration of the post-convergence burn-in stage, or 0
        when the R-hat early-stop trigger was not reached.
    """
    y_arr = _as_1d_f64(y, "y")
    m_arr = _as_2d_f64_mxn(M, "M", y_arr.shape[0])
    x_arr = None
    if X is not None:
        x_arr = _as_2d_f64(X, "X", y_arr.shape[0], allow_1d=True)

    return _call_bayesc(
        y_arr,
        m_arr,
        x_arr,
        n_iter,
        burnin,
        r2,
        df0_b,
        s0_b,
        prob_in,
        counts,
        pi,
        df0_e,
        s0_e,
        seed,
    )


class BAYES:
    def __init__(
        self,
        y: np.ndarray,
        M: np.ndarray,
        cov: np.ndarray | None = None,
        method: typing.Literal["BayesA", "BayesB", "BayesC"] = "BayesA",
        n_iter: Optional[int] = None,
        burnin: Optional[int] = None,
        r2: Optional[float] = None,
        prob_in: float = 0.5,
        counts: float = 5.0,
        pi: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        Bayesian genomic prediction using BayesA/B/C with minimal hyperparameters.

        Parameters
        ----------
        y : np.ndarray
            Phenotype vector of shape (n, 1).
        M : np.ndarray
            Marker matrix of shape (m, n) with genotypes coded as 0/1/2.
        cov : np.ndarray, optional
            Fixed-effect design matrix of shape (n, p).
        method : {"BayesA","BayesB","BayesC"}
            Bayesian model to fit.
        r2 : float, optional
            Proportion of variance explained by markers. If None, estimated
            via GBLUP (BLUP with kinship=1).
        prob_in : float
            Prior inclusion probability (BayesB/BayesC).
        counts : float
            Prior strength for inclusion probability (BayesB/BayesC).
        pi : float, optional
            Fixed marker inclusion probability for BayesB/BayesC. If omitted,
            the sampler updates the inclusion probability from active markers.

        Attributes
        ----------
        beta_hat : np.ndarray
            Posterior mean marker effects.
        alpha_hat : np.ndarray
            Posterior mean covariate effects.
        varbeta_hat : np.ndarray or float
            Posterior mean marker variances.
        varep_hat : float
            Posterior mean residual variance.
        h2_mean : float
            Posterior mean heritability.
        varh2 : float
            Posterior variance of heritability.
        rhat_h2 : float
            Split-chain R-hat of the retained posterior h2 samples.
        rhat_max_iterations : int
            Hard upper bound for R-hat monitoring and the complete chain.
        post_convergence_burnin : int
            Additional iterations discarded after the R-hat trigger.
        actual_iterations : int
            Number of MCMC updates actually performed, including any post-
            convergence burn-in iterations.
        post_burnin_start_iteration : int
            One-based first iteration of the post-convergence burn-in stage, or 0
            when the R-hat early-stop trigger was not reached.
        """
        method_map = {
            "BayesA": BayesA,
            "BayesB": BayesB,
            "BayesC": BayesC,
        }
        if method not in method_map:
            raise ValueError(f"Unsupported Bayes method: {method}")
        default_n_iter, default_burnin = bayes_mcmc_defaults(method)
        if n_iter is None:
            n_iter = int(default_n_iter)
        if burnin is None:
            burnin = int(default_burnin)

        r2_blup_pheno_scale: float | None = None
        if r2 is None:
            model = BLUP(y, M, cov=cov, kinship=1)
            r2_blup_pheno_scale = float(model.pve)
            r2 = float(r2_blup_pheno_scale)
        r2 = 0.05 if r2 <0.05 else r2; r2 = 0.95 if r2 >0.95 else r2 # optimize r2
        X = (
            np.concatenate([np.ones((M.shape[1], 1)), cov], axis=1)
            if cov is not None
            else np.ones((M.shape[1], 1))
        )
        y = y.reshape(-1, 1)

        self.method = method
        self.beta_hat: np.ndarray
        self.alpha_hat: np.ndarray
        self.varbeta_hat: np.ndarray | float | None = None
        self.varep_hat: float | None = None
        self.pve: float | None = None
        self.varpve: float | None = None
        self.pip_hat: np.ndarray | None = None
        self.rhat_h2: float = float("nan")
        self.rhat: float = float("nan")
        self.rhat_max_iterations: int = int(n_iter)
        self.post_convergence_burnin: int = int(burnin)
        self.actual_iterations: int = 0
        self.post_burnin_start_iteration: int = 0
        self.burnin_start_iteration: int = 0
        self.r2_used: float | None = float(r2)
        self.r2_blup: float | None = (
            float(r2_blup_pheno_scale) if r2_blup_pheno_scale is not None else float("nan")
        )
        self.r2_source: str = "blup_auto" if r2_blup_pheno_scale is not None else "provided"

        method_kwargs = dict(
            n_iter=n_iter,
            burnin=burnin,
            r2=float(r2),
            prob_in=prob_in,
            counts=counts,
            seed=seed,
        )
        if method in {"BayesB", "BayesC"}:
            method_kwargs["pi"] = pi
        beta, alpha, varbeta, varep, h2_mean, varh2, *diag = method_map[method](
            y,
            M,
            X,
            **method_kwargs,
        )
        self.beta_hat = beta.reshape(-1, 1);self.varbeta_hat = varbeta
        self.alpha_hat = alpha.reshape(-1, 1)
        self.varep_hat = float(varep)
        self.pve = float(h2_mean);self.varpve = float(varh2)
        diagnostics = _parse_bayes_diagnostics(
            method,
            diag,
            int(self.beta_hat.size),
        )
        pip = diagnostics["pip"]
        if isinstance(pip, np.ndarray):
            self.pip_hat = np.ascontiguousarray(pip.reshape(-1, 1), dtype=np.float64)
        self.rhat_h2 = float(diagnostics["rhat_h2"])
        self.actual_iterations = int(diagnostics["actual_iterations"])
        self.post_burnin_start_iteration = int(
            diagnostics["post_burnin_start_iteration"]
        )
        self.burnin_start_iteration = self.post_burnin_start_iteration
        self.rhat = float(self.rhat_h2)
        
    def predict(self,M:np.ndarray,cov:np.ndarray=None):
        """
        Fast solution of the mixed linear model via Brent's method.

        Parameters
        ----------
        M : np.ndarray
            Marker matrix of shape (m, n) with genotypes coded as 0/1/2.
        cov : np.ndarray, optional
            Fixed-effect design matrix of shape (n, p).
        """
        X = (
            np.concatenate([np.ones((M.shape[1], 1)), cov], axis=1)
            if cov is not None
            else np.ones((M.shape[1], 1))
        )
        return (self.beta_hat.T@M).reshape(-1,1) + X @ self.alpha_hat


bayesA = BayesA
bayesB = BayesB
bayesC = BayesC
__all__ = [
    "BAYES_MCMC_DEFAULTS",
    "bayes_mcmc_defaults",
    "BayesA",
    "BayesB",
    "BayesC",
    "BAYES",
    "bayesA",
    "bayesB",
    "bayesC",
]
