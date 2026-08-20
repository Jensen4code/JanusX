from __future__ import annotations
from typing import Optional, Tuple
import typing
import warnings
import numpy as np

from janusx.janusx import (
    bayesa as _bayesa,
    bayesb as _bayesb,
    bayesc as _bayesc,
    bayesr as _bayesr,
)
from janusx.pyBLUP.mlm import BLUP

_BAYESA_MIN_ABS_BETA_WARNED = False
BAYES_POSTERIOR_SAMPLE_TARGET = 1000

# Production GS defaults. The first value is the hard upper bound for the
# R-hat monitoring phase; the second is the fixed posterior sample target.
BAYES_MCMC_DEFAULTS: dict[str, tuple[int, int]] = {
    "BayesA": (3000, 1000),
    "BayesB": (3000, 1000),
    "BayesC": (3000, 1000),
    "BayesR": (3000, 1000),
}


def bayes_mcmc_defaults(method: str) -> tuple[int, int]:
    """Return ``(rhat_max_iter, posterior_samples)`` defaults."""
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

    The current BayesA kernel appends ``rhat, actual_iterations,
    convergence_iteration, posterior_samples``. BayesB/BayesC retain a
    12-element Rust ABI (PyO3 tuple support is limited to 12 items), while the
    Python wrappers append the fixed posterior count. The parser also accepts
    older installed extensions and avoids the p==1 scalar PIP ambiguity.
    """
    tail = list(diag)
    pip_item: object | None = None
    rhat_item: object | None = None
    actual_item: object | None = None
    convergence_item: object | None = None
    posterior_item: object | None = None
    method_key = str(method).strip().lower()
    if method_key in {"bayesb", "bayesc"}:
        if len(tail) >= 7:
            pip_item, rhat_item, actual_item, convergence_item, posterior_item = tail[-5:]
        elif len(tail) >= 6:
            # Previous ABI: pip, rhat, actual_iterations, post-burn-in start.
            pip_item, rhat_item, actual_item, convergence_item = tail[-4:]
        elif len(tail) >= 4:
            pip_item, rhat_item = tail[-2:]
        elif len(tail) >= 3:
            # Pre-R-hat B/C ABI: prob_in, n_active, pip.
            pip_item = tail[-1]
    elif len(tail) >= 4:
        rhat_item, actual_item, convergence_item, posterior_item = tail[-4:]
    elif len(tail) >= 3:
        # Previous ABI: rhat, actual_iterations, post-burn-in start.
        rhat_item, actual_item, convergence_item = tail[-3:]
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
    convergence = _scalar_from_native(convergence_item, integer=True)
    posterior = _scalar_from_native(posterior_item, integer=True)
    posterior_count = (
        int(posterior)
        if posterior is not None
        else (BAYES_POSTERIOR_SAMPLE_TARGET if actual is not None and actual > 0 else 0)
    )
    return {
        "pip": pip,
        "rhat_h2": float(rhat) if rhat is not None else float("nan"),
        "actual_iterations": int(actual) if actual is not None else 0,
        "convergence_iteration": int(convergence) if convergence is not None else 0,
        "posterior_samples": posterior_count,
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
    prior_ss_e: Optional[float],
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
    int,
]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    thin = 1
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")
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
    if prior_ss_e is not None and prior_ss_e <= 0.0:
        raise ValueError("prior_ss_e must be > 0")
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
        prior_ss_e=prior_ss_e,
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
    prior_ss_e: Optional[float],
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
    int,
]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    thin = 1
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")
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
    if prior_ss_e is not None and prior_ss_e <= 0.0:
        raise ValueError("prior_ss_e must be > 0")
    if not (0.0 < prob_in < 1.0):
        raise ValueError("prob_in must be in (0, 1)")
    if counts < 0.0:
        raise ValueError("counts must be >= 0")
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")

    fixed_pi = _validate_fixed_pi(pi)
    native_result = _bayesb(
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
        prior_ss_e=prior_ss_e,
        seed=seed,
    )
    # The native B/C ABI remains a 12-item tuple for PyO3 compatibility;
    # expose the fixed posterior count at the Python API boundary.
    return (*native_result, BAYES_POSTERIOR_SAMPLE_TARGET)


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
    prior_ss_e: Optional[float],
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
    int,
]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    thin = 1
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")
    if not (0.0 < r2 < 1.0):
        raise ValueError("r2 must be in (0, 1)")
    if df0_b <= 0.0 or df0_e <= 0.0:
        raise ValueError("df0_b and df0_e must be > 0")
    if s0_b is not None and s0_b <= 0.0:
        raise ValueError("s0_b must be > 0")
    if prior_ss_e is not None and prior_ss_e <= 0.0:
        raise ValueError("prior_ss_e must be > 0")
    if not (0.0 < prob_in < 1.0):
        raise ValueError("prob_in must be in (0, 1)")
    if counts < 0.0:
        raise ValueError("counts must be >= 0")
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")

    fixed_pi = _validate_fixed_pi(pi)
    native_result = _bayesc(
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
        prior_ss_e=prior_ss_e,
        seed=seed,
    )
    return (*native_result, BAYES_POSTERIOR_SAMPLE_TARGET)


def _call_bayesr(
    y: np.ndarray,
    m: np.ndarray,
    x: Optional[np.ndarray],
    n_iter: int,
    burnin: int,
    r2: float,
    df0_e: float,
    prior_ss_e: Optional[float],
    pi: Optional[np.ndarray],
    gamma: Optional[np.ndarray],
    df0_lambda: float,
    s0_lambda2: float,
    seed: Optional[int],
) -> dict[str, object]:
    n_iter = int(n_iter)
    burnin = int(burnin)
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")
    if not (0.0 < r2 < 1.0):
        raise ValueError("r2 must be in (0, 1)")
    if df0_e <= 0.0 or df0_lambda <= 0.0:
        raise ValueError("df0_e and df0_lambda must be > 0")
    if prior_ss_e is not None and prior_ss_e <= 0.0:
        raise ValueError("prior_ss_e must be > 0")
    if s0_lambda2 <= 0.0:
        raise ValueError("s0_lambda2 must be > 0")
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")

    def _prior(values: Optional[np.ndarray], default: tuple[float, ...], name: str) -> np.ndarray:
        if values is None:
            return np.ascontiguousarray(np.asarray(default, dtype=np.float64))
        arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
        if arr.size != 4 or not np.all(np.isfinite(arr)):
            raise ValueError(f"BayesR {name} must contain exactly four finite values")
        return arr

    pi_arr = _prior(pi, (0.90, 0.06, 0.03, 0.01), "pi")
    gamma_arr = _prior(gamma, (0.0, 0.01, 0.1, 1.0), "gamma")
    if np.any(pi_arr <= 0.0):
        raise ValueError("BayesR pi values must be > 0")
    if abs(float(gamma_arr[0])) > 1e-15 or np.any(gamma_arr[1:] <= 0.0):
        raise ValueError("BayesR gamma must start with 0 and have positive non-spike values")

    return dict(
        _bayesr(
            y=y,
            m=m,
            x=x,
            n_iter=n_iter,
            burnin=burnin,
            thin=1,
            r2=float(r2),
            df0_e=float(df0_e),
            prior_ss_e=prior_ss_e,
            pi=pi_arr,
            gamma=gamma_arr,
            df0_lambda=float(df0_lambda),
            s0_lambda2=float(s0_lambda2),
            seed=seed,
        )
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
    prior_ss_e: Optional[float] = None,
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
        Maximum iterations used for R-hat monitoring. After convergence, the
        sampler collects exactly 1000 posterior samples; without convergence,
        it collects a fallback 1000 samples after this limit.
    burnin : int, default=1000
        Deprecated compatibility argument; the production sampler no longer
        performs a second burn-in stage.
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
    prior_ss_e : float, optional
        Residual prior sum-of-squares ``nu_0 * S_0^2``; if None, derived from
        data. This is not ``S_0^2`` alone.
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
        Split-chain R-hat of the retained posterior h2 samples.
    actual_iterations : int
        Number of MCMC updates actually performed.
    convergence_iteration : int
        One-based iteration at which R-hat reached the stability threshold, or
        0 when the fallback window was used.
    posterior_samples : int
        Number of posterior samples retained (normally exactly 1000).
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
        prior_ss_e,
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
    prior_ss_e: Optional[float] = None,
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
        Maximum iterations used for R-hat monitoring, followed by a fixed
        1000-sample posterior window.
    burnin : int, default=1000
        Deprecated compatibility argument; ignored by the production sampler.
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
    prior_ss_e : float, optional
        Residual prior sum-of-squares ``nu_0 * S_0^2``; if None, derived from
        data. This is not ``S_0^2`` alone.
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
        Number of MCMC updates actually performed.
    convergence_iteration : int
        One-based iteration at which R-hat reached the stability threshold, or
        0 when the fallback window was used.
    posterior_samples : int
        Number of posterior samples retained (normally exactly 1000).
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
        prior_ss_e,
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
    prior_ss_e: Optional[float] = None,
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
        Maximum iterations used for R-hat monitoring, followed by a fixed
        1000-sample posterior window.
    burnin : int, default=1000
        Deprecated compatibility argument; ignored by the production sampler.
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
    prior_ss_e : float, optional
        Residual prior sum-of-squares ``nu_0 * S_0^2``; if None, derived from
        data. This is not ``S_0^2`` alone.
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
        Number of MCMC updates actually performed.
    convergence_iteration : int
        One-based iteration at which R-hat reached the stability threshold, or
        0 when the fallback window was used.
    posterior_samples : int
        Number of posterior samples retained (normally exactly 1000).
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
        prior_ss_e,
        seed,
    )


def BayesR(
    y: np.ndarray,
    M: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_iter: int = 3000,
    burnin: int = 1000,
    r2: float = 0.5,
    pi: Optional[np.ndarray] = None,
    gamma: Optional[np.ndarray] = None,
    df0_lambda: float = 1.0,
    s0_lambda2: float = 1.0,
    df0_e: float = 5.0,
    prior_ss_e: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    int,
    int,
    int,
]:
    """Fit the four-component BayesR mixture model.

    The default prior is ``pi=(.90,.06,.03,.01)`` and
    ``gamma=(0,.01,.1,1)``.  ``pi`` and ``gamma`` are function-level
    parameters; the GS CLI keeps these defaults and does not expose separate
    command-line flags.

    ``prior_ss_e`` is the residual prior sum-of-squares ``nu_0 * S_0^2``;
    it is not ``S_0^2`` alone.

    Returns ``(beta, alpha, varbeta, vare, h2, varh2, pip,
    component_prob, pi_mean, sigma_lambda2, rhat_h2, actual_iterations,
    convergence_iteration, posterior_samples)``.
    ``component_prob`` has shape ``(n_markers, 4)`` and uses Rao--Blackwell
    probabilities averaged over the retained posterior samples.
    """
    y_arr = _as_1d_f64(y, "y")
    m_arr = _as_2d_f64_mxn(M, "M", y_arr.shape[0])
    x_arr = None
    if X is not None:
        x_arr = _as_2d_f64(X, "X", y_arr.shape[0], allow_1d=True)
    result = _call_bayesr(
        y_arr,
        m_arr,
        x_arr,
        n_iter,
        burnin,
        r2,
        df0_e,
        prior_ss_e,
        pi,
        gamma,
        df0_lambda,
        s0_lambda2,
        seed,
    )
    return (
        np.ascontiguousarray(result["beta"], dtype=np.float64).reshape(-1),
        np.ascontiguousarray(result["alpha"], dtype=np.float64).reshape(-1),
        np.ascontiguousarray(result["varbeta"], dtype=np.float64).reshape(-1),
        float(result["vare"]),
        float(result["h2_mean"]),
        float(result["var_h2"]),
        np.ascontiguousarray(result["pip"], dtype=np.float64).reshape(-1),
        np.ascontiguousarray(result["component_prob"], dtype=np.float32),
        np.ascontiguousarray(result["pi"], dtype=np.float64).reshape(-1),
        float(result["sigma_lambda2"]),
        float(result["rhat_h2"]),
        int(result["actual_iterations"]),
        int(result["convergence_iteration"]),
        int(result["posterior_samples"]),
    )


class BAYES:
    def __init__(
        self,
        y: np.ndarray,
        M: np.ndarray,
        cov: np.ndarray | None = None,
        method: typing.Literal["BayesA", "BayesB", "BayesC", "BayesR"] = "BayesA",
        n_iter: Optional[int] = None,
        burnin: Optional[int] = None,
        r2: Optional[float] = None,
        prob_in: float = 0.5,
        counts: float = 5.0,
        pi: Optional[float | np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """
        Bayesian genomic prediction using BayesA/B/C/R with minimal hyperparameters.

        Parameters
        ----------
        y : np.ndarray
            Phenotype vector of shape (n, 1).
        M : np.ndarray
            Marker matrix of shape (m, n) with genotypes coded as 0/1/2.
        cov : np.ndarray, optional
            Fixed-effect design matrix of shape (n, p).
        method : {"BayesA","BayesB","BayesC","BayesR"}
            Bayesian model to fit.
        r2 : float, optional
            Proportion of variance explained by markers. If None, estimated
            via GBLUP (BLUP with kinship=1).
        prob_in : float
            Prior inclusion probability (BayesB/BayesC).
        counts : float
            Prior strength for inclusion probability (BayesB/BayesC).
        pi : float or array-like, optional
            Fixed marker inclusion probability for BayesB/BayesC. For BayesR,
            this is the four-component initial mixture prior and is updated
            by the Dirichlet step. If omitted, BayesR uses ``(.90,.06,.03,.01)``.
        gamma : array-like, optional
            BayesR component variance multipliers. Defaults to ``(0,.01,.1,1)``.

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
            Hard upper bound for R-hat monitoring.
        convergence_iteration : int
            Iteration at which R-hat converged, or 0 when the fallback window
            was used.
        posterior_samples : int
            Number of posterior samples retained after monitoring.
        actual_iterations : int
            Number of MCMC updates actually performed.
        """
        method_map = {
            "BayesA": BayesA,
            "BayesB": BayesB,
            "BayesC": BayesC,
            "BayesR": BayesR,
        }
        if method not in method_map:
            raise ValueError(f"Unsupported Bayes method: {method}")
        default_n_iter, default_posterior_samples = bayes_mcmc_defaults(method)
        if n_iter is None:
            n_iter = int(default_n_iter)
        if burnin is None:
            # Keep the public argument for compatibility. The native sampler
            # uses its fixed 1000-sample posterior target instead.
            burnin = int(default_posterior_samples)

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
        self.component_prob_hat: np.ndarray | None = None
        self.pi_hat: np.ndarray | None = None
        self.sigma_lambda2_hat: float | None = None
        self.rhat_h2: float = float("nan")
        self.rhat: float = float("nan")
        self.rhat_max_iterations: int = int(n_iter)
        self.posterior_sample_target: int = BAYES_POSTERIOR_SAMPLE_TARGET
        self.actual_iterations: int = 0
        self.convergence_iteration: int = 0
        self.posterior_samples: int = 0
        self.r2_used: float | None = float(r2)
        self.r2_blup: float | None = (
            float(r2_blup_pheno_scale) if r2_blup_pheno_scale is not None else float("nan")
        )
        self.r2_source: str = "blup_auto" if r2_blup_pheno_scale is not None else "provided"

        method_kwargs = dict(
            n_iter=n_iter,
            burnin=burnin,
            r2=float(r2),
            seed=seed,
        )
        if method != "BayesR":
            method_kwargs["prob_in"] = prob_in
            method_kwargs["counts"] = counts
        if method in {"BayesB", "BayesC"}:
            method_kwargs["pi"] = pi
        if method == "BayesR":
            method_kwargs["pi"] = pi
            method_kwargs["gamma"] = gamma
            (
                beta,
                alpha,
                varbeta,
                varep,
                h2_mean,
                varh2,
                pip,
                component_prob,
                pi_mean,
                sigma_lambda2,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
                posterior_samples,
            ) = method_map[method](y, M, X, **method_kwargs)
            self.component_prob_hat = np.ascontiguousarray(
                np.asarray(component_prob, dtype=np.float32), dtype=np.float32
            )
            self.pi_hat = np.ascontiguousarray(
                np.asarray(pi_mean, dtype=np.float64).reshape(-1), dtype=np.float64
            )
            self.sigma_lambda2_hat = float(sigma_lambda2)
            diag = []
        else:
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
        if method == "BayesR":
            diagnostics = {
                "pip": np.ascontiguousarray(np.asarray(pip, dtype=np.float64).reshape(-1)),
                "rhat_h2": float(rhat_h2),
                "actual_iterations": int(actual_iterations),
                "convergence_iteration": int(convergence_iteration),
                "posterior_samples": int(posterior_samples),
            }
        else:
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
        self.convergence_iteration = int(diagnostics["convergence_iteration"])
        self.posterior_samples = int(diagnostics["posterior_samples"])
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
bayesR = BayesR
__all__ = [
    "BAYES_MCMC_DEFAULTS",
    "BAYES_POSTERIOR_SAMPLE_TARGET",
    "bayes_mcmc_defaults",
    "BayesA",
    "BayesB",
    "BayesC",
    "BayesR",
    "BAYES",
    "bayesA",
    "bayesB",
    "bayesC",
    "bayesR",
]
