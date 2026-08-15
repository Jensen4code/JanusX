"""Shared phenotype preprocessing helpers for GWAS and GS."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri


def inverse_normal_transform(
    phenotype: pd.DataFrame,
    *,
    logger: Any = None,
) -> pd.DataFrame:
    """Apply a per-trait rank-based inverse normal transformation.

    Finite observations in each trait are assigned average ranks for ties and
    mapped with ``ndtri((rank - 0.5) / n)``.  Missing and non-finite values
    remain missing.  The returned frame keeps the original index, columns, and
    DataFrame attributes, but all transformed trait columns are float-valued.
    """
    if not isinstance(phenotype, pd.DataFrame):
        raise TypeError("inverse_normal_transform expects a pandas DataFrame")

    if phenotype.attrs.get("phenotype_transform") == "inverse_normal_rank_average":
        return phenotype.copy()

    out = phenotype.copy()
    out.attrs = dict(getattr(phenotype, "attrs", {}))
    out.attrs["phenotype_transform"] = "inverse_normal_rank_average"

    for column_idx in range(int(out.shape[1])):
        column = out.columns[column_idx]
        values = pd.to_numeric(out.iloc[:, column_idx], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=True,
        )
        finite = np.isfinite(values)
        n_finite = int(np.count_nonzero(finite))
        if n_finite == 0:
            out.iloc[:, column_idx] = np.full(values.shape, np.nan, dtype=np.float64)
            continue

        transformed = np.full(values.shape, np.nan, dtype=np.float64)
        ranks = (
            pd.Series(values[finite], dtype=np.float64)
            .rank(method="average")
            .to_numpy(dtype=np.float64)
        )
        probabilities = (ranks - 0.5) / float(n_finite)
        transformed[finite] = ndtri(probabilities)
        out.iloc[:, column_idx] = transformed

        if logger is not None and int(np.unique(values[finite]).size) <= 1:
            logger.warning(
                "Inverse-normal transformation produced a constant trait "
                f"for '{column}' because it has at most one unique finite value."
            )

    return out
