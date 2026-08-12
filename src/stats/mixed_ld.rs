use nalgebra::{DMatrix, SVD};
use numpy::ndarray::{Array1, Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;
use std::time::Instant;

use crate::blas::{rust_sgemm_backend_tag, BlasThreadGuard};

const PROJECTED_DIAG_REL_TOL: f64 = 64.0 * f64::EPSILON;

#[derive(Debug)]
struct EffectiveLdResult {
    r: DMatrix<f64>,
    valid_indices: Vec<usize>,
    fixed_effect_columns: usize,
    fixed_effect_rank: usize,
    rank_tolerance: f64,
    min_projected_diag: f64,
    rotation_seconds: f64,
    svd_seconds: f64,
    residual_seconds: f64,
    gram_seconds: f64,
    total_seconds: f64,
}

fn validate_finite_matrix(matrix: &DMatrix<f64>, name: &str) -> Result<(), String> {
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            if !matrix[(row, col)].is_finite() {
                return Err(format!(
                    "{name} contains non-finite value at ({row}, {col})"
                ));
            }
        }
    }
    Ok(())
}

fn validate_finite_vector(values: &[f64], name: &str) -> Result<(), String> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{name} contains non-finite value at index {index}"));
        }
    }
    Ok(())
}

fn fvlmm_effective_ld_spectral_core(
    genotypes: &DMatrix<f64>,
    eigvals: &[f64],
    u_t: &DMatrix<f64>,
    fixed_effects: &DMatrix<f64>,
    lambda_null: f64,
    rcond: Option<f64>,
) -> Result<EffectiveLdResult, String> {
    let total_started = Instant::now();
    let snp_count = genotypes.nrows();
    let sample_count = genotypes.ncols();
    let fixed_effect_columns = fixed_effects.ncols();

    if snp_count == 0 || sample_count == 0 {
        return Err("genotypes must have at least one SNP row and sample column".to_string());
    }
    if eigvals.len() != sample_count {
        return Err(format!(
            "eigvals length mismatch: expected {}, got {}",
            sample_count,
            eigvals.len()
        ));
    }
    if u_t.nrows() != sample_count || u_t.ncols() != sample_count {
        return Err(format!(
            "u_t shape mismatch: expected ({sample_count}, {sample_count}), got ({}, {})",
            u_t.nrows(),
            u_t.ncols()
        ));
    }
    if fixed_effects.nrows() != sample_count || fixed_effect_columns == 0 {
        return Err(format!(
            "fixed_effects shape mismatch or zero columns: expected ({sample_count}, q>0), got ({}, {})",
            fixed_effects.nrows(),
            fixed_effect_columns
        ));
    }
    if !lambda_null.is_finite() || lambda_null <= 0.0 {
        return Err("lambda_null must be finite and > 0".to_string());
    }

    validate_finite_matrix(genotypes, "genotypes")?;
    validate_finite_vector(eigvals, "eigvals")?;
    validate_finite_matrix(u_t, "u_t")?;
    validate_finite_matrix(fixed_effects, "fixed_effects")?;

    let rcond_value = match rcond {
        Some(value) if value.is_finite() && value > 0.0 => value,
        Some(_) => return Err("rcond must be finite and > 0".to_string()),
        None => (sample_count.max(fixed_effect_columns) as f64) * f64::EPSILON,
    };

    let rotation_started = Instant::now();
    let mut whitening = Vec::with_capacity(sample_count);
    for (index, &eigval) in eigvals.iter().enumerate() {
        let variance = eigval + lambda_null;
        if !variance.is_finite() || variance <= 0.0 {
            return Err(format!(
                "eigvals[{index}] + lambda_null must be finite and > 0"
            ));
        }
        let scale = 1.0 / variance.sqrt();
        if !scale.is_finite() {
            return Err(format!(
                "whitening scale for eigvals[{index}] is non-finite"
            ));
        }
        whitening.push(scale);
    }

    // `u_t` is row-major U^T.  The two products below implement
    // Gw = G U diag(1/sqrt(s + lambda)) and
    // Cw = diag(1/sqrt(s + lambda)) U^T C.
    let u = u_t.transpose();
    let mut gw = genotypes * &u;
    let mut cw = u_t * fixed_effects;
    for sample in 0..sample_count {
        let scale = whitening[sample];
        for snp in 0..snp_count {
            gw[(snp, sample)] *= scale;
        }
        for column in 0..fixed_effect_columns {
            cw[(sample, column)] *= scale;
        }
    }
    let rotation_seconds = rotation_started.elapsed().as_secs_f64();

    let svd_started = Instant::now();
    let svd = SVD::try_new(cw, true, false, 5.0 * f64::EPSILON, 0)
        .ok_or_else(|| "rank-revealing SVD did not converge".to_string())?;
    let singular_values = svd.singular_values;
    let max_singular = singular_values.iter().copied().fold(0.0_f64, f64::max);
    if !max_singular.is_finite() || max_singular <= 0.0 {
        return Err("fixed_effects has rank zero after whitening".to_string());
    }
    let rank_tolerance = rcond_value * max_singular;
    if !rank_tolerance.is_finite() || rank_tolerance <= 0.0 {
        return Err("rank tolerance is non-finite or non-positive".to_string());
    }
    let fixed_effect_rank = singular_values
        .iter()
        .copied()
        .filter(|&singular| singular.is_finite() && singular > rank_tolerance)
        .count();
    if fixed_effect_rank == 0 {
        return Err("fixed_effects has rank zero after whitening".to_string());
    }
    if fixed_effect_rank == sample_count {
        return Err(
            "fixed_effects span the full whitened sample space; no projected genotype signal remains"
                .to_string(),
        );
    }
    let u_covariates = svd
        .u
        .ok_or_else(|| "rank-revealing SVD did not produce left singular vectors".to_string())?;
    let svd_seconds = svd_started.elapsed().as_secs_f64();

    let residual_started = Instant::now();
    // Keep only Gw projected into the retained fixed-effect column space.  The
    // full n x n P matrix is never formed.
    let mut projected = DMatrix::<f64>::from_element(snp_count, fixed_effect_rank, f64::NAN);
    for snp in 0..snp_count {
        for rank_column in 0..fixed_effect_rank {
            let mut value = 0.0_f64;
            for sample in 0..sample_count {
                value += gw[(snp, sample)] * u_covariates[(sample, rank_column)];
            }
            if value.is_finite() {
                projected[(snp, rank_column)] = value;
            }
        }
    }

    // Form residualized genotype rows explicitly, rather than recovering
    // their squared norms by subtracting two nearly equal quadratic forms.
    // The latter loses a small but resolvable projected signal to cancellation.
    let mut residuals = DMatrix::<f64>::zeros(snp_count, sample_count);
    let mut projected_diag = vec![f64::NAN; snp_count];
    let mut max_projected_diag = 0.0_f64;
    for snp in 0..snp_count {
        let mut diagonal = 0.0_f64;
        let mut row_finite = true;
        for sample in 0..sample_count {
            let mut value = gw[(snp, sample)];
            for rank_column in 0..fixed_effect_rank {
                value -= projected[(snp, rank_column)] * u_covariates[(sample, rank_column)];
            }
            let squared = value * value;
            if !value.is_finite() || !squared.is_finite() {
                row_finite = false;
                break;
            }
            diagonal += squared;
            if !diagonal.is_finite() {
                row_finite = false;
                break;
            }
            residuals[(snp, sample)] = value;
        }
        if row_finite && diagonal.is_finite() {
            projected_diag[snp] = diagonal;
            max_projected_diag = max_projected_diag.max(diagonal);
        }
    }
    if !max_projected_diag.is_finite() || max_projected_diag <= 0.0 {
        return Err("projected genotype diagonals are all non-finite or zero".to_string());
    }

    let diagonal_tolerance = PROJECTED_DIAG_REL_TOL * max_projected_diag;
    let valid_indices: Vec<usize> = projected_diag
        .iter()
        .enumerate()
        .filter_map(|(index, &diagonal)| {
            (diagonal.is_finite() && diagonal > diagonal_tolerance).then_some(index)
        })
        .collect();
    if valid_indices.is_empty() {
        return Err("no SNP rows have a finite, non-negligible projected diagonal".to_string());
    }

    let min_projected_diag = valid_indices
        .iter()
        .map(|&index| projected_diag[index])
        .fold(f64::INFINITY, f64::min);
    let residual_seconds = residual_started.elapsed().as_secs_f64();

    let gram_started = Instant::now();
    let valid_count = valid_indices.len();
    let mut r = DMatrix::<f64>::zeros(valid_count, valid_count);
    for (left, &left_index) in valid_indices.iter().enumerate() {
        let left_scale = projected_diag[left_index].sqrt();
        for (right, &right_index) in valid_indices.iter().enumerate().skip(left) {
            let value = if left == right {
                1.0_f64
            } else {
                let right_scale = projected_diag[right_index].sqrt();
                let mut total = 0.0_f64;
                for sample in 0..sample_count {
                    let left_value = residuals[(left_index, sample)] / left_scale;
                    let right_value = residuals[(right_index, sample)] / right_scale;
                    let product = left_value * right_value;
                    if !product.is_finite() {
                        return Err(format!("effective LD is non-finite at ({left}, {right})"));
                    }
                    total += product;
                    if !total.is_finite() {
                        return Err(format!("effective LD is non-finite at ({left}, {right})"));
                    }
                }
                total
            };
            if !value.is_finite() {
                return Err(format!("effective LD is non-finite at ({left}, {right})"));
            }
            r[(left, right)] = value;
            r[(right, left)] = value;
        }
    }
    let gram_seconds = gram_started.elapsed().as_secs_f64();
    let total_seconds = total_started.elapsed().as_secs_f64();

    Ok(EffectiveLdResult {
        r,
        valid_indices,
        fixed_effect_columns,
        fixed_effect_rank,
        rank_tolerance,
        min_projected_diag,
        rotation_seconds,
        svd_seconds,
        residual_seconds,
        gram_seconds,
        total_seconds,
    })
}

fn array2_to_row_major<'py>(array: ArrayView2<'py, f64>) -> Vec<f64> {
    let (rows, columns) = (array.shape()[0], array.shape()[1]);
    let mut values = Vec::with_capacity(rows.saturating_mul(columns));
    for row in 0..rows {
        for column in 0..columns {
            values.push(array[(row, column)]);
        }
    }
    values
}

#[pyfunction]
#[pyo3(signature = (genotypes, eigvals, u_t, fixed_effects, lambda_null, rcond=None, threads=1))]
pub fn fvlmm_effective_ld_spectral_f64<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    eigvals: PyReadonlyArray1<'py, f64>,
    u_t: PyReadonlyArray2<'py, f64>,
    fixed_effects: PyReadonlyArray2<'py, f64>,
    lambda_null: f64,
    rcond: Option<f64>,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let genotypes_view = genotypes.as_array();
    let eigvals_view = eigvals.as_array();
    let u_t_view = u_t.as_array();
    let fixed_effects_view = fixed_effects.as_array();

    let genotypes_rows = genotypes_view.shape()[0];
    let genotypes_columns = genotypes_view.shape()[1];
    let u_t_rows = u_t_view.shape()[0];
    let u_t_columns = u_t_view.shape()[1];
    let fixed_effect_rows = fixed_effects_view.shape()[0];
    let fixed_effect_columns = fixed_effects_view.shape()[1];

    let genotypes_matrix = DMatrix::from_row_slice(
        genotypes_rows,
        genotypes_columns,
        &array2_to_row_major(genotypes_view),
    );
    let eigvals_vec: Vec<f64> = eigvals_view.iter().copied().collect();
    let u_t_matrix = DMatrix::from_row_slice(u_t_rows, u_t_columns, &array2_to_row_major(u_t_view));
    let fixed_effects_matrix = DMatrix::from_row_slice(
        fixed_effect_rows,
        fixed_effect_columns,
        &array2_to_row_major(fixed_effects_view),
    );

    let _blas_guard = BlasThreadGuard::enter(threads.max(1));
    let requested_threads = threads.max(1);
    let using_threads = requested_threads;
    let result = py
        .detach(|| {
            fvlmm_effective_ld_spectral_core(
                &genotypes_matrix,
                &eigvals_vec,
                &u_t_matrix,
                &fixed_effects_matrix,
                lambda_null,
                rcond,
            )
        })
        .map_err(PyValueError::new_err)?;

    let valid_count = result.valid_indices.len();
    let r_matrix = &result.r;
    let r_values: Vec<f64> = (0..valid_count)
        .flat_map(|row| (0..valid_count).map(move |column| r_matrix[(row, column)]))
        .collect();
    let r_array = Array2::from_shape_vec((valid_count, valid_count), r_values)
        .map_err(|error| PyValueError::new_err(format!("invalid effective LD shape: {error}")))?;
    let valid_indices: Vec<i64> = result
        .valid_indices
        .iter()
        .map(|&index| {
            i64::try_from(index)
                .map_err(|_| PyValueError::new_err("valid SNP index exceeds int64 range"))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let out = PyDict::new(py);
    out.set_item("r", PyArray2::from_owned_array(py, r_array).into_bound())?;
    out.set_item(
        "valid_indices",
        PyArray1::from_owned_array(py, Array1::from_vec(valid_indices)).into_bound(),
    )?;
    out.set_item("fixed_effect_columns", result.fixed_effect_columns)?;
    out.set_item("fixed_effect_rank", result.fixed_effect_rank)?;
    out.set_item("rank_tolerance", result.rank_tolerance)?;
    out.set_item("min_projected_diag", result.min_projected_diag)?;
    out.set_item("backend", rust_sgemm_backend_tag())?;
    out.set_item("requested_threads", requested_threads)?;
    out.set_item("using_threads", using_threads)?;
    out.set_item("rotation_seconds", result.rotation_seconds)?;
    out.set_item("svd_seconds", result.svd_seconds)?;
    out.set_item("residual_seconds", result.residual_seconds)?;
    out.set_item("gram_seconds", result.gram_seconds)?;
    out.set_item("total_seconds", result.total_seconds)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use super::fvlmm_effective_ld_spectral_core;

    fn rotation_u_t(n: usize) -> DMatrix<f64> {
        let mut out = DMatrix::<f64>::identity(n, n);
        let angle = 0.37_f64;
        let (sin, cos) = angle.sin_cos();
        out[(0, 0)] = cos;
        out[(0, 1)] = sin;
        out[(1, 0)] = -sin;
        out[(1, 1)] = cos;
        out
    }

    fn direct_reference(
        genotypes: &DMatrix<f64>,
        eigvals: &[f64],
        u_t: &DMatrix<f64>,
        fixed_effects: &DMatrix<f64>,
        lambda_null: f64,
    ) -> DMatrix<f64> {
        let n = eigvals.len();
        let v_inv = u_t.transpose()
            * DMatrix::from_diagonal(&DVector::from_iterator(
                n,
                eigvals.iter().map(|&s| 1.0_f64 / (s + lambda_null)),
            ))
            * u_t;
        let c_v_inv_c = fixed_effects.transpose() * &v_inv * fixed_effects;
        let p = &v_inv
            - &v_inv
                * fixed_effects
                * c_v_inv_c
                    .try_inverse()
                    .expect("full-rank reference inverse")
                * fixed_effects.transpose()
                * &v_inv;
        let q = genotypes * p * genotypes.transpose();
        let scales: Vec<f64> = (0..q.nrows()).map(|i| q[(i, i)].sqrt()).collect();
        DMatrix::from_fn(q.nrows(), q.ncols(), |i, j| {
            if i == j {
                1.0
            } else {
                q[(i, j)] / (scales[i] * scales[j])
            }
        })
    }

    fn assert_matrix_close(actual: &DMatrix<f64>, expected: &DMatrix<f64>, tol: f64) {
        assert_eq!(actual.shape(), expected.shape());
        for i in 0..actual.nrows() {
            for j in 0..actual.ncols() {
                assert!(
                    (actual[(i, j)] - expected[(i, j)]).abs() <= tol,
                    "mismatch at ({i}, {j}): actual={:.16e}, expected={:.16e}",
                    actual[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    #[test]
    fn effective_ld_matches_direct_p_projection() {
        let n = 6;
        let genotypes = DMatrix::from_row_slice(
            3,
            n,
            &[
                0.0, 1.0, 2.0, 1.0, 0.5, 2.0, // SNP 0
                2.0, 0.0, 1.0, 2.0, 1.0, 0.0, // SNP 1
                1.5, 2.0, 0.5, 0.0, 2.0, 1.0, // SNP 2
            ],
        );
        let eigvals = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0];
        let u_t = rotation_u_t(n);
        let fixed_effects = DMatrix::from_column_slice(
            n,
            2,
            &[
                1.0, -1.0, 1.0, 0.5, 1.0, 0.0, // intercept
                -1.0, -0.5, 0.25, 1.0, 1.5, 2.0, // covariate
            ],
        );
        let lambda_null = 0.4_f64;

        let actual = fvlmm_effective_ld_spectral_core(
            &genotypes,
            &eigvals,
            &u_t,
            &fixed_effects,
            lambda_null,
            None,
        )
        .expect("full-rank effective LD");
        let expected = direct_reference(&genotypes, &eigvals, &u_t, &fixed_effects, lambda_null);

        assert_eq!(actual.valid_indices, vec![0, 1, 2]);
        assert_eq!(actual.fixed_effect_columns, 2);
        assert_eq!(actual.fixed_effect_rank, 2);
        assert!(actual.rank_tolerance.is_finite());
        assert!(actual.rank_tolerance > 0.0);
        assert_matrix_close(&actual.r, &expected, 1.0e-10);
        for i in 0..actual.r.nrows() {
            assert_eq!(actual.r[(i, i)], 1.0);
            for j in 0..actual.r.ncols() {
                assert!(actual.r[(i, j)].is_finite());
                assert_eq!(actual.r[(i, j)], actual.r[(j, i)]);
            }
        }
    }

    #[test]
    fn duplicate_covariate_uses_rank_one_projection() {
        let n = 4;
        let genotypes = DMatrix::from_row_slice(2, n, &[0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0]);
        let eigvals = [0.0, 0.0, 0.0, 0.0];
        let u_t = DMatrix::<f64>::identity(n, n);
        let duplicated = DMatrix::from_row_slice(n, 2, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let reduced = DMatrix::from_column_slice(n, 1, &[1.0, 1.0, 1.0, 1.0]);

        let actual =
            fvlmm_effective_ld_spectral_core(&genotypes, &eigvals, &u_t, &duplicated, 1.0, None)
                .expect("duplicate covariate effective LD");
        let expected = direct_reference(&genotypes, &eigvals, &u_t, &reduced, 1.0);

        assert_eq!(actual.fixed_effect_columns, 2);
        assert_eq!(actual.fixed_effect_rank, 1);
        assert_matrix_close(&actual.r, &expected, 1.0e-10);
    }

    #[test]
    fn near_collinear_covariate_is_finite_and_deterministic() {
        let n = 6;
        let x = [-1.0, -0.5, 0.25, 1.0, 1.5, 2.0];
        let mut covariates = Vec::with_capacity(n * 2);
        for &value in &x {
            covariates.push(value);
            covariates.push(value + 1.0e-14);
        }
        let genotypes = DMatrix::from_row_slice(
            3,
            n,
            &[
                0.0, 1.0, 2.0, 1.0, 0.5, 2.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.5, 2.0, 0.5, 0.0,
                2.0, 1.0,
            ],
        );
        let eigvals = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0];
        let u_t = rotation_u_t(n);
        let covariates = DMatrix::from_row_slice(n, 2, &covariates);

        let first =
            fvlmm_effective_ld_spectral_core(&genotypes, &eigvals, &u_t, &covariates, 0.4, None)
                .expect("near-collinear effective LD");
        let second =
            fvlmm_effective_ld_spectral_core(&genotypes, &eigvals, &u_t, &covariates, 0.4, None)
                .expect("repeat near-collinear effective LD");

        assert!(first.fixed_effect_rank >= 1);
        assert!(first.fixed_effect_rank <= 2);
        assert_eq!(first.valid_indices, second.valid_indices);
        assert_eq!(first.r.as_slice(), second.r.as_slice());
        assert!(first.r.iter().all(|value| value.is_finite()));
        for i in 0..first.r.nrows() {
            assert_eq!(first.r[(i, i)], 1.0);
            for j in 0..first.r.ncols() {
                assert_eq!(first.r[(i, j)], first.r[(j, i)]);
            }
        }
    }

    #[test]
    fn tiny_but_resolvable_projected_signal_is_retained() {
        let genotypes = DMatrix::from_row_slice(1, 2, &[1.0, 1.0 + 1.0e-8]);
        let eigvals = [0.0, 0.0];
        let u_t = DMatrix::<f64>::identity(2, 2);
        let fixed_effects = DMatrix::from_column_slice(2, 1, &[1.0, 1.0]);

        let actual =
            fvlmm_effective_ld_spectral_core(&genotypes, &eigvals, &u_t, &fixed_effects, 1.0, None)
                .expect("small nonzero projected signal should remain usable");

        assert_eq!(actual.valid_indices, vec![0]);
        assert!(actual.min_projected_diag > 0.0);
        assert_eq!(actual.r[(0, 0)], 1.0);
    }

    #[test]
    fn full_rank_fixed_effects_have_no_projected_signal() {
        let angle = 0.37_f64;
        let (sin, cos) = angle.sin_cos();
        let genotypes = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        let eigvals = [0.0, 0.0];
        let u_t = DMatrix::<f64>::identity(2, 2);
        let fixed_effects = DMatrix::from_row_slice(2, 2, &[cos, -sin, sin, cos]);

        let err =
            fvlmm_effective_ld_spectral_core(&genotypes, &eigvals, &u_t, &fixed_effects, 1.0, None)
                .expect_err("full-rank fixed effects must leave no projected signal");
        assert!(err.contains("full whitened sample space"));
    }

    #[test]
    fn degenerate_projected_rows_are_removed_with_original_indices() {
        let genotypes = DMatrix::from_row_slice(
            3,
            4,
            &[
                1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0e-8, 2.0e-8, 1.0e-8,
            ],
        );
        let eigvals = [0.0, 0.0, 0.0, 0.0];
        let u_t = DMatrix::<f64>::identity(4, 4);
        let fixed_effects = DMatrix::from_column_slice(4, 1, &[1.0, 1.0, 1.0, 1.0]);

        let actual =
            fvlmm_effective_ld_spectral_core(&genotypes, &eigvals, &u_t, &fixed_effects, 1.0, None)
                .expect("one non-degenerate projected row should remain");

        assert_eq!(actual.valid_indices, vec![1]);
        assert_eq!(actual.r.shape(), (1, 1));
        assert_eq!(actual.r[(0, 0)], 1.0);
    }

    #[test]
    fn non_finite_or_rank_zero_fixed_effects_are_rejected() {
        let genotypes = DMatrix::from_row_slice(1, 3, &[0.0, 1.0, 2.0]);
        let eigvals = [0.0, 0.0, 0.0];
        let u_t = DMatrix::<f64>::identity(3, 3);

        let mut non_finite = DMatrix::from_column_slice(3, 1, &[1.0, 1.0, 1.0]);
        non_finite[(1, 0)] = f64::NAN;
        let err =
            fvlmm_effective_ld_spectral_core(&genotypes, &eigvals, &u_t, &non_finite, 1.0, None)
                .expect_err("non-finite fixed effects must fail");
        assert!(err.contains("finite"));

        let zero = DMatrix::<f64>::zeros(3, 1);
        let err = fvlmm_effective_ld_spectral_core(&genotypes, &eigvals, &u_t, &zero, 1.0, None)
            .expect_err("rank-zero fixed effects must fail");
        assert!(err.contains("rank"));
    }
}
