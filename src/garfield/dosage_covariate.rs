//! Fixed-effect projection helpers for the dosage interaction backend.
//!
//! The projector is deliberately independent from the genotype representation.
//! A single QR factorization of `Z = [1, covariates]` is reused by all three
//! dosage backends; each backend only has to provide raw moments and
//! `Q' * (g1 * g2)` for a pair.

use nalgebra::DMatrix;

const RANK_TOL: f64 = 1.0e-12;

#[derive(Clone, Debug)]
pub(crate) struct CovariateProjector {
    n: usize,
    rank: usize,
    /// Thin Q from a column-pivoted Householder QR, stored row-major.
    q_rows: Vec<f64>,
    /// Upper-triangular R from the same factorization, stored row-major.
    r_upper: Vec<f64>,
    /// Maps an original fixed-effect column to its position in the
    /// pivoted QR solve.  The decomposition is `Z P = Q R`.
    inverse_permutation: Vec<usize>,
}

impl CovariateProjector {
    /// Construct a full-rank fixed-effect projector with an implicit intercept.
    ///
    /// `covariates` is row-major `(n, q_cov)`.  A rank-deficient design is
    /// rejected rather than projected with a pseudoinverse: silently dropping
    /// a duplicate dummy/PC would change the model contract.
    pub(crate) fn from_covariates(
        covariates: &[f64],
        n: usize,
        n_covariates: usize,
    ) -> Result<Self, String> {
        if n == 0 {
            return Err("covariate projection requires n > 0".to_string());
        }
        let expected = n
            .checked_mul(n_covariates)
            .ok_or_else(|| "covariate design size overflow".to_string())?;
        if covariates.len() != expected {
            return Err(format!(
                "covariate length mismatch: got {}, expected {}",
                covariates.len(),
                expected
            ));
        }
        if covariates.iter().any(|value| !value.is_finite()) {
            return Err("covariates contain non-finite values".to_string());
        }

        let n_fixed = n_covariates
            .checked_add(1)
            .ok_or_else(|| "fixed-effect column count overflow".to_string())?;
        if n <= n_fixed {
            return Err(format!(
                "covariate projection requires n > fixed-effect columns; got n={n}, columns={n_fixed}"
            ));
        }

        let design_len = n
            .checked_mul(n_fixed)
            .ok_or_else(|| "fixed-effect design size overflow".to_string())?;
        let mut design = vec![0.0_f64; design_len];
        for row in 0..n {
            let dst = row * n_fixed;
            design[dst] = 1.0;
            if n_covariates > 0 {
                let src = row * n_covariates;
                design[dst + 1..dst + n_fixed]
                    .copy_from_slice(&covariates[src..src + n_covariates]);
            }
        }

        let matrix = DMatrix::from_row_slice(n, n_fixed, &design);
        let qr = matrix.col_piv_qr();
        let r = qr.r();
        let diag_scale = (0..n_fixed)
            .map(|column| r[(column, column)].abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let tolerance = RANK_TOL * (n.max(n_fixed) as f64) * diag_scale;
        let rank = (0..n_fixed)
            .filter(|&column| r[(column, column)].abs() > tolerance)
            .count();
        if rank != n_fixed {
            return Err(format!(
                "covariate design is rank-deficient: rank={rank}, columns={n_fixed}, tolerance={tolerance:.3e}"
            ));
        }

        let q = qr.q();
        let mut q_rows = vec![0.0_f64; design_len];
        for row in 0..n {
            for column in 0..n_fixed {
                q_rows[row * n_fixed + column] = q[(row, column)];
            }
        }
        let r = qr.r();
        let r_len = n_fixed
            .checked_mul(n_fixed)
            .ok_or_else(|| "fixed-effect R factor size overflow".to_string())?;
        let mut r_upper = vec![0.0_f64; r_len];
        for row in 0..n_fixed {
            for column in 0..n_fixed {
                r_upper[row * n_fixed + column] = r[(row, column)];
            }
        }
        let mut pivoted_columns = DMatrix::from_fn(n_fixed, 1, |row, _| row as f64);
        qr.p().permute_rows(&mut pivoted_columns);
        let mut inverse_permutation = vec![0usize; n_fixed];
        for (pivoted_position, original_column) in pivoted_columns.column(0).iter().enumerate() {
            let original_column = *original_column as usize;
            if original_column >= n_fixed {
                return Err("invalid QR column permutation".to_string());
            }
            inverse_permutation[original_column] = pivoted_position;
        }
        Ok(Self {
            n,
            rank,
            q_rows,
            r_upper,
            inverse_permutation,
        })
    }

    #[inline]
    pub(crate) fn n_samples(&self) -> usize {
        self.n
    }

    #[inline]
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub(crate) fn q_row(&self, row: usize) -> Option<&[f64]> {
        (row < self.n).then(|| &self.q_rows[row * self.rank..(row + 1) * self.rank])
    }

    #[inline]
    pub(crate) fn q_coordinates(&self, values: &[f64], out: &mut [f64]) -> Result<(), String> {
        if values.len() != self.n {
            return Err(format!(
                "vector length mismatch: got {}, expected {}",
                values.len(),
                self.n
            ));
        }
        if out.len() != self.rank {
            return Err(format!(
                "projection coordinate length mismatch: got {}, expected {}",
                out.len(),
                self.rank
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("projection vector contains non-finite values".to_string());
        }
        out.fill(0.0);
        for row in 0..self.n {
            let value = values[row];
            let q_row = &self.q_rows[row * self.rank..(row + 1) * self.rank];
            for column in 0..self.rank {
                out[column] += q_row[column] * value;
            }
        }
        Ok(())
    }

    /// Return the inner product after removing the fixed-effect subspace.
    #[inline]
    pub(crate) fn projected_inner_product(
        &self,
        left: &[f64],
        right: &[f64],
    ) -> Result<f64, String> {
        if right.len() != self.n {
            return Err(format!(
                "vector length mismatch: got {}, expected {}",
                right.len(),
                self.n
            ));
        }
        let mut left_q = vec![0.0_f64; self.rank];
        let mut right_q = vec![0.0_f64; self.rank];
        self.q_coordinates(left, &mut left_q)?;
        self.q_coordinates(right, &mut right_q)?;
        let raw = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let projected = raw
            - left_q
                .iter()
                .zip(right_q.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        if projected.is_finite() {
            Ok(projected)
        } else {
            Err("projected inner product is non-finite".to_string())
        }
    }

    #[inline]
    pub(crate) fn project_vector(&self, values: &[f64], out: &mut [f64]) -> Result<(), String> {
        if out.len() != self.n {
            return Err(format!(
                "projected vector length mismatch: got {}, expected {}",
                out.len(),
                self.n
            ));
        }
        let mut coordinates = vec![0.0_f64; self.rank];
        self.q_coordinates(values, &mut coordinates)?;
        for row in 0..self.n {
            let q_row = &self.q_rows[row * self.rank..(row + 1) * self.rank];
            let projection = q_row
                .iter()
                .zip(coordinates.iter())
                .map(|(q, coefficient)| q * coefficient)
                .sum::<f64>();
            out[row] = values[row] - projection;
        }
        Ok(())
    }

    /// Solve the fixed-effect coefficients for a vector in the span of the
    /// design columns.  This is the back-substitution counterpart of the
    /// stored column-pivoted QR and deliberately does not use a pseudoinverse.
    #[inline]
    pub(crate) fn solve_fixed_effects(
        &self,
        values: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if values.len() != self.n {
            return Err(format!(
                "vector length mismatch: got {}, expected {}",
                values.len(),
                self.n
            ));
        }
        if out.len() != self.rank {
            return Err(format!(
                "fixed-effect coefficient length mismatch: got {}, expected {}",
                out.len(),
                self.rank
            ));
        }
        let mut rhs = vec![0.0_f64; self.rank];
        self.q_coordinates(values, &mut rhs)?;
        for row in (0..self.rank).rev() {
            let tail = ((row + 1)..self.rank)
                .map(|column| self.r_upper[row * self.rank + column] * rhs[column])
                .sum::<f64>();
            let diagonal = self.r_upper[row * self.rank + row];
            if !(diagonal.abs() > RANK_TOL) || !diagonal.is_finite() {
                return Err(format!("rank-deficient fixed-effect solve at pivot {row}"));
            }
            rhs[row] = (rhs[row] - tail) / diagonal;
        }
        for original_column in 0..self.rank {
            out[original_column] = rhs[self.inverse_permutation[original_column]];
        }
        if out.iter().any(|value| !value.is_finite()) {
            return Err("fixed-effect coefficients are non-finite".to_string());
        }
        Ok(())
    }
}
