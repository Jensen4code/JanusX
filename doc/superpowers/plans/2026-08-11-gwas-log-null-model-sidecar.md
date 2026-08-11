# GWAS Log Null-Model Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a versioned FvLMM null-model sidecar in each GWAS log, match and validate it from PostGWAS, and use the corresponding GRM projection to build numerically stable mixed-model LD for SuSiE fine-mapping.

**Architecture:** A focused Python module owns sidecar serialization, fingerprints, discovery, and matching. GWAS emits one deterministic JSON block after each FvLMM result is finalized; PostGWAS reconstructs the exact sample/fixed-effect context and delegates effective-LD construction to a Rust/PyO3 numerical kernel. The kernel works on the whitened scale and projects fixed effects with a rank-revealing SVD, never an explicit inverse.

**Tech Stack:** Python 3.13, pandas, NumPy, Rust, PyO3/numpy, nalgebra SVD, pytest, cargo test, maturin, and susieR 0.14.2 for external numerical comparison.

## Global Constraints

- The production CLI remains `jx postgwas ... -finemap susie`; any explicit GWAS-log override is development-only.
- Mixed-model fine-mapping never silently falls back to raw genotype LD.
- Missing, malformed, ambiguous, unsupported, or mismatched metadata emits a warning and skips fine-mapping without failing unrelated PostGWAS work.
- A skipped run publishes no new PIP or CS files and warns that existing same-prefix outputs may be stale.
- The first supported mixed-model method is FvLMM; exact LMM is skipped until a common-null score Z is available.
- Effective-LD code does not materialize dense `P` and does not explicitly invert `C^T V^-1 C`.
- Exact or near rank-deficient fixed effects use a rank-revealing SVD/pseudoinverse with an explicit relative tolerance.
- Existing `-mem` accounting covers genotype, GRM/eigensystem, effective-LD, and SuSiE workspace.
- Existing allele alignment, clumping/fold restoration, numeric formatting, atomic publication, backup, and rollback remain intact.
- Generated Rice6048 results and other experiment artifacts stay local and are never committed or pushed.

## File Structure

- Create `python/janusx/assoc/null_model_sidecar.py`: schema, fingerprints, ordered-ID hashing, parser, discovery, and matching.
- Create `src/stats/mixed_ld.rs`: FvLMM whitening, rank-revealing projection, normalization, and PyO3 function.
- Modify `src/lib.rs`: register the mixed-LD module and Python function.
- Modify `python/janusx/assoc/workflow.py`: create run source metadata and pass sidecar context.
- Modify `python/janusx/assoc/workflow_model_stream.py`: emit FvLMM records after successful TSV publication.
- Modify `python/janusx/script/postgwas.py`: validate sidecars, reconstruct FvLMM state, build effective LD, and implement warning-and-skip behavior.
- Create `test/test_gwas_null_model_sidecar.py`: codec, matching, emission, and skip tests.
- Create `test/test_postgwas_fvlmm_effective_ld.py`: numerical kernel, reconstruction, integration, and regression tests.

---

### Task 1: Versioned Sidecar Schema and Matcher

**Files:**
- Create: `python/janusx/assoc/null_model_sidecar.py`
- Create: `test/test_gwas_null_model_sidecar.py`

**Interfaces:**
- Produces: `FileFingerprintV1`, `GwasNullModelSidecarV1`, `fingerprint_file()`, `hash_ordered_sample_ids()`, `serialize_sidecar_block()`, `parse_sidecar_blocks()`, and `find_matching_sidecar()`.
- Consumes: Python standard library only.

- [ ] **Step 1: Write failing codec and hash tests**

```python
from janusx.assoc.null_model_sidecar import (
    GwasNullModelSidecarV1,
    hash_ordered_sample_ids,
    parse_sidecar_blocks,
    serialize_sidecar_block,
)

def test_sidecar_roundtrip_is_deterministic(tmp_path):
    result = tmp_path / "run.Trait.fvlmm.tsv"
    result.write_text("chrom\tpos\tbeta\tse\n1\t10\t1\t1\n")
    record = make_record(result)
    block = serialize_sidecar_block(record)
    assert block == serialize_sidecar_block(record)
    assert parse_sidecar_blocks("prefix\n" + block + "suffix\n") == [record]

def test_ordered_sample_hash_distinguishes_order():
    assert hash_ordered_sample_ids(["A", "BC"]) != hash_ordered_sample_ids(["BC", "A"])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `mamba run -n jxfu pytest -q test/test_gwas_null_model_sidecar.py`

Expected: collection fails because `janusx.assoc.null_model_sidecar` is absent.

- [ ] **Step 3: Implement immutable v1 schema and deterministic codec**

Define these sentinels and frozen data classes:

```python
SIDECAR_SCHEMA_V1 = "janusx.gwas-null-model/v1"
SIDECAR_BEGIN_V1 = "[JANUSX_GWAS_NULL_MODEL_V1_BEGIN]"
SIDECAR_END_V1 = "[JANUSX_GWAS_NULL_MODEL_V1_END]"

@dataclass(frozen=True)
class FileFingerprintV1:
    canonical_path: str
    basename: str
    size_bytes: int
    sha256: str

@dataclass(frozen=True)
class GwasNullModelSidecarV1:
    schema: str
    created_at: str
    janusx_version: str
    result: FileFingerprintV1
    model: str
    trait: str
    z_columns: tuple[str, str]
    effective_snp_count: int
    genotype_prefix: str
    genotype_files: tuple[FileFingerprintV1, ...]
    phenotype_file: FileFingerprintV1
    phenotype_id_column: str
    phenotype_trait_column: str
    covariate_file: FileFingerprintV1 | None
    covariate_columns: tuple[str, ...]
    kinship_file: FileFingerprintV1
    kinship_id_file: FileFingerprintV1
    kinship_format: str
    kinship_shape: tuple[int, int]
    sample_count: int
    sample_order_sha256: str
    lambda_null: float
    sigma_g2: float | None
    sigma_e2: float | None
    pve: float
    grm_trace_mean: float
    fixed_effect_columns: tuple[str, ...]
    genotype_filters: dict[str, object]
    allele_coding: str
```

Serialize sorted compact JSON. Hash ordered IDs using an eight-byte big-endian
length prefix followed by UTF-8 bytes. Reject incomplete sentinels, duplicate
JSON keys, unsupported schemas, invalid hashes, non-finite required numbers,
and non-positive dimensions with `SidecarFormatError`.

- [ ] **Step 4: Add exact-path, moved-file, ambiguity, and malformed-block tests**

```python
def test_matching_prefers_exact_canonical_path(tmp_path):
    result = write_result(tmp_path / "run.Trait.fvlmm.tsv")
    record = make_record(result)
    assert find_matching_sidecar(result, [record]) == record

def test_matching_accepts_moved_result_by_name_size_and_sha256(tmp_path):
    original = write_result(tmp_path / "old" / "run.Trait.fvlmm.tsv")
    record = make_record(original)
    moved = tmp_path / "new" / original.name
    moved.parent.mkdir()
    shutil.copyfile(original, moved)
    assert find_matching_sidecar(moved, [record]) == record

def test_matching_rejects_two_equivalent_candidates(tmp_path):
    result = write_result(tmp_path / "run.Trait.fvlmm.tsv")
    record = make_record(result)
    with pytest.raises(SidecarAmbiguous):
        find_matching_sidecar(result, [record, record])

def test_parser_rejects_incomplete_and_duplicate_key_json():
    with pytest.raises(SidecarFormatError):
        parse_sidecar_blocks(SIDECAR_BEGIN_V1 + "\n{}\n")
    duplicate = '{"schema":"janusx.gwas-null-model/v1","schema":"x"}'
    with pytest.raises(SidecarFormatError):
        parse_sidecar_blocks(SIDECAR_BEGIN_V1 + "\n" + duplicate + "\n" + SIDECAR_END_V1)
```

Define `write_result(path)` in the test file to create parent directories, write
the fixed four-column TSV used in Step 1, and return `path`. Define
`make_record(result)` in the test file to instantiate every required v1 field
with finite tiny-file values and `fingerprint_file(result)`; do not add test-only
constructors to production classes.

`find_matching_sidecar()` returns one record or raises typed `SidecarNotFound`,
`SidecarAmbiguous`, or `SidecarFormatError`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `mamba run -n jxfu pytest -q test/test_gwas_null_model_sidecar.py`

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add python/janusx/assoc/null_model_sidecar.py test/test_gwas_null_model_sidecar.py
git commit -m "feat(gwas): define null-model log sidecar schema"
```

---

### Task 2: Emit FvLMM Sidecars After Result Publication

**Files:**
- Modify: `python/janusx/assoc/null_model_sidecar.py`
- Modify: `python/janusx/assoc/workflow.py`
- Modify: `python/janusx/assoc/workflow_model_stream.py`
- Modify: `test/test_gwas_null_model_sidecar.py`

**Interfaces:**
- Consumes: Task 1 schema and codec.
- Produces: `GwasSidecarRunContext`, `build_fvlmm_sidecar()`, and `emit_sidecar_to_file_log()`.

- [ ] **Step 1: Write failing builder and publication-order tests**

```python
def test_builder_records_fitted_null_and_sample_hash(tmp_path):
    record = build_fvlmm_sidecar(
        context=make_context(tmp_path), result_file=make_result(tmp_path),
        trait="Trait", sample_ids=["I2", "I1"], lambda_null=0.081,
        sigma_g2=2.0, sigma_e2=0.162, pve=0.95,
        grm_trace_mean=1.7, effective_snp_count=12,
    )
    assert record.model == "fvlmm"
    assert record.sample_order_sha256 == hash_ordered_sample_ids(["I2", "I1"])

def test_sidecar_is_not_emitted_before_result_exists(tmp_path, caplog):
    missing = tmp_path / "missing.fvlmm.tsv"
    with pytest.raises(FileNotFoundError):
        build_fvlmm_sidecar(
            context=make_context(tmp_path), result_file=missing, trait="Trait",
            sample_ids=["I1"], lambda_null=1.0, sigma_g2=1.0,
            sigma_e2=1.0, pve=0.5, grm_trace_mean=1.0,
            effective_snp_count=1,
        )
    assert SIDECAR_BEGIN_V1 not in caplog.text

def test_multiple_traits_emit_distinct_records(tmp_path):
    first = build_test_record(tmp_path, "Height")
    second = build_test_record(tmp_path, "Yield")
    parsed = parse_sidecar_blocks(
        serialize_sidecar_block(first) + serialize_sidecar_block(second)
    )
    assert [record.trait for record in parsed] == ["Height", "Yield"]
    assert parsed[0].result != parsed[1].result
```

Define `make_context()` to create the tiny referenced input files and
`build_test_record()` to create a distinct result TSV and call the builder with
finite fixed null values. These are test-only helpers in the same file.

- [ ] **Step 2: Run tests and verify RED**

Run: `mamba run -n jxfu pytest -q test/test_gwas_null_model_sidecar.py -k 'builder or emitted or multiple_traits'`

Expected: tests fail because builder/emitter interfaces are absent.

- [ ] **Step 3: Implement run context and cached streamed fingerprints**

```python
@dataclass
class GwasSidecarRunContext:
    genotype_prefix: Path
    phenotype_file: Path
    phenotype_id_column: str
    covariate_file: Path | None
    covariate_columns: tuple[str, ...]
    kinship_file: Path
    kinship_id_file: Path
    genotype_filters: dict[str, object]
    fingerprint_cache: dict[Path, FileFingerprintV1] = field(default_factory=dict)
```

Stream SHA-256 in fixed chunks and reuse cached BED/BIM/FAM, phenotype,
covariate, GRM, and GRM-ID fingerprints across traits.

- [ ] **Step 4: Wire both maintained streaming publication paths**

Add `null_sidecar_context: GwasSidecarRunContext | None` to
`run_chunked_gwas_lmm_lm()` and `run_chunked_gwas_streaming_shared()`. Immediately
after `_finalize_gwas_result_tsv()` succeeds for FvLMM, build from the actual
aligned sample IDs and fitted `mod.lbd_null`, `sigma_g2_null`, `sigma_e2_null`,
`pve`, and `trace_mean`. Write the block only through `_gwas_report_logger()`.
Do not emit for empty/failed results or other models.

- [ ] **Step 5: Run syntax, emission, and FvLMM route tests**

```bash
mamba run -n jxfu python -m py_compile python/janusx/assoc/null_model_sidecar.py python/janusx/assoc/workflow.py python/janusx/assoc/workflow_model_stream.py
mamba run -n jxfu pytest -q test/test_gwas_null_model_sidecar.py test/test_fvlmm_scan_optimization.py
```

Expected: one parseable record per successful FvLMM result and all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add python/janusx/assoc/null_model_sidecar.py python/janusx/assoc/workflow.py python/janusx/assoc/workflow_model_stream.py test/test_gwas_null_model_sidecar.py test/test_fvlmm_scan_optimization.py
git commit -m "feat(gwas): record FvLMM null model in log"
```

---

### Task 3: PostGWAS Discovery and Warning-and-Skip Semantics

**Files:**
- Modify: `python/janusx/assoc/null_model_sidecar.py`
- Modify: `python/janusx/script/postgwas.py`
- Modify: `test/test_gwas_null_model_sidecar.py`

**Interfaces:**
- Consumes: Task 1 parser/matcher.
- Produces: `discover_matching_sidecar()`, `validate_sidecar_dependencies()`, `FineMapSkip`, and optional fine-mapping output.

- [ ] **Step 1: Write failing discovery and skip tests**

```python
def test_discovers_match_across_same_directory_logs(tmp_path):
    result = write_result(tmp_path / "run.Trait.fvlmm.tsv")
    record = make_record(result)
    (tmp_path / "unrelated.gwas.log").write_text("ordinary log\n")
    (tmp_path / "run.gwas.log").write_text(serialize_sidecar_block(record))
    assert discover_matching_sidecar(result) == record

@pytest.mark.parametrize("failure", ["missing", "ambiguous", "malformed", "result_hash"])
def test_sidecar_failure_warns_and_publishes_nothing(tmp_path, caplog, failure):
    assert run_test_finemap(tmp_path, failure=failure) is None
    assert not (tmp_path / "out.susie.pip.tsv").exists()
    assert not (tmp_path / "out.susie.cs.tsv").exists()
    assert "fine-mapping skipped" in caplog.text
    assert "pre-existing" in caplog.text
```

Define `run_test_finemap(tmp_path, failure)` in this task as a test-only wrapper
that creates a valid tiny FvLMM result/sidecar, mutates exactly the requested
failure condition, invokes `_run_postgwas_susie_finemap()`, and returns its
optional output path. It must create no initial PIP/CS files in this parametrized
test so publication assertions are unambiguous.

- [ ] **Step 2: Run tests and verify RED**

Run: `mamba run -n jxfu pytest -q test/test_gwas_null_model_sidecar.py -k 'discovers or publishes_nothing'`

Expected: PostGWAS still enters raw LD or raises a fatal error.

- [ ] **Step 3: Implement deterministic same-directory discovery and validation**

Scan sorted `*.gwas.log`, parse complete blocks, and require exactly one match.
Validate result, BED/BIM/FAM, phenotype, covariate, GRM, and GRM-ID fingerprints;
verify supplied `-bfile`, model, schema, and allele convention. Raise
`FineMapSkip(reason)` only for expected compatibility failures.

- [ ] **Step 4: Catch skips outside paired-output publication**

Change `_run_postgwas_susie_finemap()` to return `Optional[str]`. Catch only
`FineMapSkip`, warn with the exact reason, state that no new PIP/CS was generated
and prior outputs may be stale, remove only current-run temporary files, and
return `None`. Keep programming/publication exceptions fatal.

- [ ] **Step 5: Verify skip and ordinary PostGWAS behavior**

```bash
mamba run -n jxfu pytest -q test/test_gwas_null_model_sidecar.py
mamba run -n jxfu python -m py_compile python/janusx/script/postgwas.py
```

Expected: all expected failures warn and skip; ordinary paths remain passing.

- [ ] **Step 6: Commit Task 3**

```bash
git add python/janusx/assoc/null_model_sidecar.py python/janusx/script/postgwas.py test/test_gwas_null_model_sidecar.py
git commit -m "feat(postgwas): match mixed-model GWAS sidecars"
```

---

### Task 4: Rust Effective-LD Kernel with Rank-Revealing Projection

**Files:**
- Create: `src/stats/mixed_ld.rs`
- Modify: `src/lib.rs`
- Create: `test/test_postgwas_fvlmm_effective_ld.py`

**Interfaces:**
- Produces: `fvlmm_effective_ld_spectral_f64(genotypes, eigvals, u_t, fixed_effects, lambda_null, rcond=None, threads=1) -> dict`.
- Returns: `r`, `valid_indices`, `fixed_effect_columns`, `fixed_effect_rank`, `rank_tolerance`, and `min_projected_diag`.

- [ ] **Step 1: Write failing Rust tests**

```rust
#[test]
fn effective_ld_matches_direct_p_projection() { /* n=6, p=3, q=2 */ }
#[test]
fn duplicate_covariate_uses_rank_one_projection() { /* C=[1,1] */ }
#[test]
fn near_collinear_covariate_is_finite_and_deterministic() { /* x2=x1+1e-14 */ }
#[test]
fn non_finite_or_rank_zero_fixed_effects_are_rejected() { /* typed Err */ }
```

The tiny direct reference may materialize `P`. Assert LD within `1e-10`, rank,
finite values, symmetry, and unit diagonal.

- [ ] **Step 2: Run Rust tests and verify RED**

Run: `DYLD_LIBRARY_PATH="$(mamba run -n jxfu python -c 'import sys; print(sys.prefix)')/lib" PYO3_PYTHON="$(mamba run -n jxfu python -c 'import sys; print(sys.executable)')" cargo test mixed_ld --lib`

Expected: compile failure because `mixed_ld` is absent.

- [ ] **Step 3: Implement whitening and SVD column-space projection**

Use:

```text
Gw = G U diag(1/sqrt(s+lambda))
Cw = diag(1/sqrt(s+lambda)) U^T C
Cw = Uc S V^T
tol = rcond.unwrap_or(max(n,q)*eps) * max(S)
rank = count(S_i > tol)
Q = Gw Gw^T - (Gw Uc_rank)(Gw Uc_rank)^T
R = diag(Q)^(-1/2) Q diag(Q)^(-1/2)
```

This is the Moore-Penrose projection and avoids squaring the condition number.
Use existing BLAS/thread guards. Exclude genotype rows with a non-finite or
scale-negligible projected diagonal and return their surviving original indices.

- [ ] **Step 4: Export PyO3 and add Python wrapper tests**

Register the module/function in `src/lib.rs`. Add:

```python
def test_rust_effective_ld_reports_rank_deficiency():
    g = np.array([[0, 1, 2, 1], [2, 1, 0, 1]], dtype=np.float64)
    c = np.column_stack([np.ones(4), np.ones(4)])
    out = jxrs.fvlmm_effective_ld_spectral_f64(
        g, np.zeros(4), np.eye(4), c, 1.0, threads=1
    )
    assert out["fixed_effect_columns"] == 2
    assert out["fixed_effect_rank"] == 1
    assert np.isfinite(np.asarray(out["r"])).all()

def test_rust_effective_ld_rejects_bad_shapes_and_nonfinite_input():
    with pytest.raises(ValueError, match="eigvals"):
        jxrs.fvlmm_effective_ld_spectral_f64(
            np.ones((2, 4)), np.ones(3), np.eye(4), np.ones((4, 1)), 1.0
        )
    bad = np.ones((4, 1)); bad[2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        jxrs.fvlmm_effective_ld_spectral_f64(
            np.ones((2, 4)), np.ones(4), np.eye(4), bad, 1.0
        )

def test_rust_effective_ld_is_thread_deterministic():
    g = np.arange(24, dtype=np.float64).reshape(4, 6) % 3
    args = (g, np.linspace(0.1, 1.0, 6), np.eye(6), np.ones((6, 1)), 0.5)
    one = jxrs.fvlmm_effective_ld_spectral_f64(*args, threads=1)
    four = jxrs.fvlmm_effective_ld_spectral_f64(*args, threads=4)
    np.testing.assert_allclose(one["r"], four["r"], atol=1e-12, rtol=1e-12)
```

- [ ] **Step 5: Build and verify Rust/Python tests**

```bash
cargo fmt --all -- --check
DYLD_LIBRARY_PATH="$(mamba run -n jxfu python -c 'import sys; print(sys.prefix)')/lib" PYO3_PYTHON="$(mamba run -n jxfu python -c 'import sys; print(sys.executable)')" cargo test mixed_ld --lib
PREFIX="$(mamba run -n jxfu python -c 'import sys; print(sys.prefix)')"
CONDA_PREFIX="$PREFIX" PIP_BREAK_SYSTEM_PACKAGES=1 "$PREFIX/bin/python" -m maturin develop --release --locked --features python-extension
mamba run -n jxfu pytest -q test/test_postgwas_fvlmm_effective_ld.py -k rust
```

Expected: formatting, Rust tests, build, and wrapper tests all pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/stats/mixed_ld.rs src/lib.rs test/test_postgwas_fvlmm_effective_ld.py
git commit -m "feat(finemap): add rank-aware FvLMM LD kernel"
```

---

### Task 5: Reconstruct Exact FvLMM Samples and Fixed Effects

**Files:**
- Modify: `python/janusx/assoc/null_model_sidecar.py`
- Modify: `python/janusx/script/postgwas.py`
- Modify: `test/test_postgwas_fvlmm_effective_ld.py`

**Interfaces:**
- Consumes: validated Task 3 sidecar and Task 4 kernel.
- Produces: `FvLMMFineMapContext`, `_postgwas_reconstruct_fvlmm_context()`, and `_postgwas_build_fvlmm_effective_ld()`.

- [ ] **Step 1: Write failing reconstruction tests**

Use FAM order `I1,I2,I3,I4`, shuffled phenotype rows, one missing phenotype,
and duplicate/near-collinear covariates:

```python
def test_reconstructs_sample_order_and_fixed_effects(tmp_path):
    record = write_reconstruction_fixture(tmp_path)
    context = _postgwas_reconstruct_fvlmm_context(record)
    assert context.sample_ids.tolist() == ["I1", "I3", "I4"]
    assert context.fixed_effects[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert hash_ordered_sample_ids(context.sample_ids) == context.sidecar.sample_order_sha256

def test_sample_hash_mismatch_skips(tmp_path):
    record = replace(write_reconstruction_fixture(tmp_path), sample_order_sha256="0" * 64)
    with pytest.raises(FineMapSkip, match="sample order"):
        _postgwas_reconstruct_fvlmm_context(record)

def test_grm_id_mismatch_skips(tmp_path):
    record = write_reconstruction_fixture(tmp_path)
    id_path = Path(record.kinship_id_file.canonical_path)
    id_path.write_text("I1\nI3\nUNKNOWN\n")
    record = replace(record, kinship_id_file=fingerprint_file(id_path))
    with pytest.raises(FineMapSkip, match="GRM ID"):
        _postgwas_reconstruct_fvlmm_context(record)

def test_duplicate_covariates_reach_rank_aware_kernel(tmp_path):
    context = _postgwas_reconstruct_fvlmm_context(write_reconstruction_fixture(tmp_path))
    assert context.fixed_effects.shape == (3, 3)
    np.testing.assert_allclose(context.fixed_effects[:, 1], context.fixed_effects[:, 2])
```

Define `write_reconstruction_fixture()` in this task to write all source files,
set covariate columns `batch_a,batch_b` to identical values, calculate current
fingerprints, and return a valid record whose sample hash is for `I1,I3,I4`.

- [ ] **Step 2: Run tests and verify RED**

Run: `mamba run -n jxfu pytest -q test/test_postgwas_fvlmm_effective_ld.py -k 'reconstructs or sample_hash or duplicate_covariates'`

Expected: reconstruction interfaces are absent.

- [ ] **Step 3: Implement deterministic reconstruction**

```python
@dataclass
class FvLMMFineMapContext:
    sample_ids: np.ndarray
    sample_indices_in_bfile: np.ndarray
    fixed_effects: np.ndarray
    kinship: np.ndarray
    lambda_null: float
    sidecar: GwasNullModelSidecarV1
```

Reapply GWAS phenotype/covariate missingness and FAM-order intersection. Verify
the ordered sample hash before loading the full GRM. Align GRM rows through its
ID file. Reject duplicate/missing IDs, dimension mismatches, invalid lambda, or
non-finite fixed effects with `FineMapSkip`. Preserve duplicate covariate columns
for the SVD kernel rather than deleting them heuristically.

- [ ] **Step 4: Implement memory preflight and effective-LD adapter**

Estimate GRM subset, eigensystem, genotype, rotated matrices, projected Gram,
and SuSiE arrays against `-mem` before large allocations. Then load verified
samples, eigendecompose `K + 1e-6 I` with the existing LAPACK-backed helper,
call `fvlmm_effective_ld_spectral_f64`, and subset GWAS/Z using
`valid_indices`. Log sample count, lambda, PVE, fixed-effect columns/rank/tolerance,
minimum LD eigenvalue, and condition number.

- [ ] **Step 5: Run focused reconstruction tests**

```bash
mamba run -n jxfu pytest -q test/test_postgwas_fvlmm_effective_ld.py
mamba run -n jxfu python -m py_compile python/janusx/assoc/null_model_sidecar.py python/janusx/script/postgwas.py
```

Expected: context, mismatch, memory, and rank-deficiency tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add python/janusx/assoc/null_model_sidecar.py python/janusx/script/postgwas.py test/test_postgwas_fvlmm_effective_ld.py
git commit -m "feat(postgwas): reconstruct FvLMM null projection"
```

---

### Task 6: Route SuSiE Through Effective LD

**Files:**
- Modify: `python/janusx/script/postgwas.py`
- Modify: `test/test_postgwas_fvlmm_effective_ld.py`
- Modify: `test/test_gwas_null_model_sidecar.py`

**Interfaces:**
- Consumes: Tasks 3 and 5 validated state.
- Produces: FvLMM uses effective LD; LM retains sample-matched ordinary LD; unsupported mixed models warn and skip.

- [ ] **Step 1: Write failing route and rollback tests**

```python
def test_fvlmm_calls_effective_ld_not_raw_ld(monkeypatch, route_fixture):
    calls = []
    monkeypatch.setattr(postgwas, "_postgwas_build_fvlmm_effective_ld",
                        lambda *args, **kwargs: calls.append("mixed") or route_fixture.ld_result)
    monkeypatch.setattr(postgwas, "_postgwas_build_sample_matched_raw_ld",
                        lambda *args, **kwargs: pytest.fail("raw LD called for FvLMM"))
    route_fixture.run(model="fvlmm")
    assert calls == ["mixed"]

def test_lm_keeps_sample_matched_raw_ld(monkeypatch, route_fixture):
    calls = []
    monkeypatch.setattr(postgwas, "_postgwas_build_sample_matched_raw_ld",
                        lambda *args, **kwargs: calls.append("raw") or route_fixture.ld_result)
    monkeypatch.setattr(postgwas, "_postgwas_build_fvlmm_effective_ld",
                        lambda *args, **kwargs: pytest.fail("mixed LD called for LM"))
    route_fixture.run(model="lm")
    assert calls == ["raw"]

def test_exact_lmm_warns_and_skips_without_touching_existing_outputs(route_fixture, caplog):
    before = route_fixture.output_hashes()
    assert route_fixture.run(model="lmm") is None
    assert route_fixture.output_hashes() == before
    assert "common-null score" in caplog.text

def test_effective_ld_failure_restores_paired_previous_outputs(monkeypatch, route_fixture):
    before = route_fixture.output_hashes()
    monkeypatch.setattr(postgwas, "_postgwas_build_fvlmm_effective_ld",
                        lambda *args, **kwargs: (_ for _ in ()).throw(FineMapSkip("bad LD")))
    assert route_fixture.run(model="fvlmm") is None
    assert route_fixture.output_hashes() == before
```

Define `route_fixture` as a pytest fixture with `ld_result`, `run(model)`, and
`output_hashes()` methods. It writes valid paired old outputs, a minimal matched
sidecar for the selected model, and invokes `_run_postgwas_susie_finemap()` with
fully populated argparse values.

Make the forbidden LD function raise in each route test. Hash both pre-existing
PIP/CS files before and after the rollback test.

- [ ] **Step 2: Run tests and verify RED**

Run: `mamba run -n jxfu pytest -q test/test_postgwas_fvlmm_effective_ld.py -k 'calls_effective or keeps_sample or exact_lmm or restores'`

Expected: FvLMM still calls raw LD or unsupported routes are not skipped safely.

- [ ] **Step 3: Select the model route before raw dense-LD allocation**

```python
if record.model == "fvlmm":
    locus_r, aligned = _postgwas_build_fvlmm_effective_ld(
        args=args, record=record, prepared=prepared,
        bed_indices=bed_indices, logger=logger,
    )
elif record.model in {"lmm", "lmm2", "splmm", "splmm2"}:
    raise FineMapSkip(f"{record.model} does not expose common-null score statistics")
else:
    locus_r, aligned = _postgwas_build_sample_matched_raw_ld(
        args=args, prepared=prepared, bed_indices=bed_indices, logger=logger,
    )
```

Keep allele alignment and clumping on the verified GWAS sample subset. Do not
allocate the old raw dense LD before this decision.

- [ ] **Step 4: Preserve alignment, folded SNPs, and diagnostics**

When projected-diagonal filtering removes variants, update aligned rows, Z,
BED indices, and fold dictionaries through one explicit index map. Log SuSiE
iterations, convergence, and all prior variances. Keep current PIP formatting,
CS restoration, temporary files, backup, and rollback unchanged.

- [ ] **Step 5: Run complete Python fine-mapping tests**

```bash
mamba run -n jxfu pytest -q test/test_gwas_null_model_sidecar.py test/test_postgwas_fvlmm_effective_ld.py
mamba run -n jxfu python -m py_compile python/janusx/script/postgwas.py
```

Expected: sidecar, route, rank, formatting, and rollback tests pass.

- [ ] **Step 6: Commit Task 6**

```bash
git add python/janusx/script/postgwas.py test/test_gwas_null_model_sidecar.py test/test_postgwas_fvlmm_effective_ld.py
git commit -m "feat(postgwas): use FvLMM effective LD for SuSiE"
```

---

### Task 7: Synthetic Numerical Regression and susieR Agreement

**Files:**
- Modify: `test/test_postgwas_fvlmm_effective_ld.py`

**Interfaces:**
- Consumes: completed sidecar and effective-LD route.
- Produces: committed small-data regression proving the raw-LD instability is removed.

- [ ] **Step 1: Write deterministic end-to-end regression**

Generate fixed `n<=80`, `p<=20` structured genotype, GRM, rank-deficient
covariates, and phenotype under pytest `tmp_path`. Assert:

```python
assert fit["converged"] is True
assert np.max(np.abs(fit["posterior_mean"])) < 5 * np.max(np.abs(z))
assert np.max(fit["prior_variance"]) < 100 * np.max(z * z)
assert diagnostics.fixed_effect_rank < diagnostics.fixed_effect_columns
```

Run the same Z with raw LD and require either non-convergence or a larger
summary/LD mismatch diagnostic, proving the fixture exercises this defect.

- [ ] **Step 2: Run test and verify RED**

Run: `mamba run -n jxfu pytest -q test/test_postgwas_fvlmm_effective_ld.py -k synthetic_end_to_end`

Expected: the first absent fixture/helper or effective-LD assertion fails.

- [ ] **Step 3: Add only deterministic fixture helpers**

Use a fixed NumPy RNG seed and explicit allele orientation. Keep all generated
files under `tmp_path`; add no binary or experiment fixtures to Git.

- [ ] **Step 4: Add optional susieR parity test**

Gate on `Rscript` and susieR availability. Compare:

```python
np.testing.assert_allclose(janusx_pip, susier_pip, atol=1e-6, rtol=1e-5)
np.testing.assert_allclose(janusx_pm, susier_pm, atol=1e-6, rtol=1e-5)
```

- [ ] **Step 5: Run regression and parity tests**

Run: `mamba run -n jxfu pytest -q test/test_postgwas_fvlmm_effective_ld.py`

Expected: synthetic regression passes; susieR passes in `jxfu` or reports one
explicit skip when unavailable.

- [ ] **Step 6: Commit Task 7**

```bash
git add test/test_postgwas_fvlmm_effective_ld.py
git commit -m "test(finemap): cover FvLMM LD mismatch regression"
```

---

### Task 8: Rice6048 Local Validation and Final Gate

**Files:**
- Modify only if validation exposes a source defect: files owned by the failing task.
- Do not add: `test.20260805/**`, `.janusx_test/**`, generated PIP/CS/log files, or local matrices.

**Interfaces:**
- Consumes: Tasks 1-7.
- Produces: fresh real-data evidence and final regression status.

- [ ] **Step 1: Run focused static, Rust, and Python verification**

```bash
cargo fmt --all -- --check
DYLD_LIBRARY_PATH="$(mamba run -n jxfu python -c 'import sys; print(sys.prefix)')/lib" PYO3_PYTHON="$(mamba run -n jxfu python -c 'import sys; print(sys.executable)')" cargo test mixed_ld --lib
mamba run -n jxfu pytest -q test/test_gwas_null_model_sidecar.py test/test_postgwas_fvlmm_effective_ld.py test/test_fvlmm_scan_optimization.py test/test_reml_interface.py test/test_reml_integration.py
```

Expected: formatting and all focused tests pass without unexpected skips.

- [ ] **Step 2: Generate a fresh local Rice6048 FvLMM sidecar**

```bash
JX_GWAS_PLOT=0 mamba run -n jxfu jx gwas \
  -bfile test.atlas/Rice6048 \
  -p test.20260805/test.blup.txt \
  -k test.atlas/Rice6048.cGRM.npy \
  -fvlmm -t 8 -o test.20260805/sidecar_validation
```

Expected: one Plant_height sidecar with 3273 samples, PVE near `0.9559`, and
lambda near `0.0810`.

- [ ] **Step 3: Run Rice6048 effective-LD fine-mapping**

```bash
mamba run -n jxfu jx postgwas \
  -i test.20260805/sidecar_validation.Plant_height.fvlmm.tsv \
  -o test.20260805/sidecar_validation/Rice6048.Plant_height \
  -bfile test.atlas/Rice6048 \
  -bimrange 1:38414420:38461203 \
  -finemap susie -finemap-max-iter 500 -finemap-L 5 -mem 5
```

Expected: sidecar matching succeeds, fixed-effect rank is logged, SuSiE
converges, maximum absolute posterior mean is near the diagnostic reference
`22.05` rather than `554.25`, and prior variance remains far below `1e5`.

- [ ] **Step 4: Compare emitted Z/effective R with susieR 0.14.2**

Export diagnostic arrays only under the local validation directory or `/tmp`.
Run susieR with `L=5`, `max_iter=500`, and `tol=1e-4`; apply Task 7 tolerances
to PIP and posterior mean and record both convergence/iteration counts.

- [ ] **Step 5: Run project smoke and inspect repository scope**

```bash
.agents/skills/janusx-project/scripts/check_jxfu.sh
git diff --check
git status --short --branch
git diff --stat origin/main...HEAD
```

Expected: smoke passes or only documented pre-existing failures remain; feature
source/tests/docs are scoped; no generated Rice6048 or `.janusx_test` artifact is staged.

- [ ] **Step 6: Commit a source-only correction only if validation required one**

When no correction was needed, create no empty commit. If required, stage only
the defect's source and committed test files, rerun its failing gate, and use a
specific message, for example:

```bash
git add src/stats/mixed_ld.rs python/janusx/script/postgwas.py test/test_postgwas_fvlmm_effective_ld.py
git commit -m "fix(finemap): preserve FvLMM effective-LD alignment"
```
