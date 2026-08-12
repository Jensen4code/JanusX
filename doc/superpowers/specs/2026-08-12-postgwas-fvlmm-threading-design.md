# PostGWAS FvLMM Threading and Throughput Design

## Goal

Remove the apparent single-thread stall after FvLMM fine-mapping memory
preflight while preserving exact variant identity, sample alignment, memory
limits, deterministic output order, and JanusX/susieR numerical agreement.

The motivating Rice6048 run has 5,694,922 BIM rows, approximately 169 regional
summary variants, and 3,273 FvLMM samples. The current implementation first
materializes every BIM row as Python tuples and then scans that collection once
per regional variant. That is roughly 962 million Python comparisons before
the effective-LD numerical work starts.

## Scope

The work proceeds in four ordered, independently reviewable stages:

1. Replace repeated full-BIM matching with one streaming indexed scan.
2. Add stage timing, thread-count, and backend diagnostics.
3. Route dense effective-LD matrix products through the existing controlled
   BLAS layer.
4. Parallelize only the remaining row-local work and compute the final Gram
   matrix through BLAS.

The work does not alter the statistical model, sidecar schema, clumping
threshold, SuSiE algorithm, output formats, or allele convention. It does not
add a GPU path.

## 1. Streaming BIM Identity Resolution

### Current defect

`_postgwas_read_bim_identity_rows()` returns all BIM rows, and
`_postgwas_match_prepared_variants_to_bim()` builds a full-list
comprehension for each prepared row. Besides the quadratic-like
`regional_variants × BIM_rows` cost, millions of Python tuples consume
substantial memory.

### New interface

Add a focused resolver:

```python
def _postgwas_resolve_prepared_bim_rows(
    genotype_prefix: object,
    prepared: pd.DataFrame,
) -> tuple[
    list[int],
    list[tuple[str, int, str | None, str | None, str | None]],
]:
    ...
```

It validates prepared positions and optional SNP/allele identity first, then
builds small target maps keyed by normalized chromosome and position. It scans
the BIM once in source order, recording only matching rows and their zero-based
BIM indices. After EOF it applies the existing identity rules:

- exact SNP ID is preferred when supplied;
- allele-pair identity is enforced when supplied;
- coordinate fallback is allowed only when unambiguous;
- one prepared row cannot resolve to the same BIM row as another;
- duplicate-coordinate variants remain distinct when SNP/alleles prove their
  identity;
- missing, malformed, mismatched, or ambiguous identities raise
  `FineMapSkip`.

The resolver returns selected metadata in prepared-row order, so downstream
genotype decoding and alignment retain their current contract. No full BIM
tuple list remains live.

### Complexity

- Time: `O(BIM rows + prepared rows + matching candidates)`.
- Python object memory: `O(prepared rows + matching candidates)`.
- The BIM file is still scanned sequentially because the source must preserve
  row indices for exact PLINK SNP selection.

## 2. Stage Diagnostics

The FvLMM route will log elapsed wall time immediately after each stage:

- context/sample reconstruction;
- streaming BIM target scan;
- regional genotype decode and alignment;
- GRM eigendecomposition;
- effective-LD kernel total;
- effective-LD clumping;
- SuSiE fitting.

The Rust kernel will return or expose substage diagnostics for:

- genotype/fixed-effect whitening and rotation;
- fixed-effect rank-revealing SVD;
- row residualization and projected-diagonal filtering;
- normalized LD Gram construction.

Every relevant log record includes requested/using threads. EVD and
effective-LD logs also include the selected backend. Diagnostics must not print
per-SNP messages or materially change runtime.

Timing values are observational metadata only and are excluded from result
hashes and statistical output files.

## 3. BLAS-Controlled Dense Products

### Thread ownership

Each dense BLAS region is wrapped by `BlasThreadGuard::enter(threads)`.
During a BLAS call, Rayon does not run nested parallel work. During Rayon
row-local work, BLAS is held to one thread where applicable. This avoids
oversubscription.

On macOS the selected backend is Accelerate CPU. The threading guard toggles
Accelerate's threading mode but does not invoke GPU compute. On Linux the same
interface controls OpenBLAS.

### Products

The following products move from nalgebra operator multiplication or explicit
dot loops to the repository's checked f64 GEMM interface:

```text
Gw = G U
Cw = U^T C
projected = Gw Uc
R_raw = residuals residuals^T
```

Whitening scales are then applied deterministically along columns/rows. The
rank-revealing SVD remains the established nalgebra SVD unless measurement
shows it dominates; with one or a few fixed effects it is not expected to be a
meaningful bottleneck.

The implementation must use the existing backend-dispatch conventions rather
than adding direct platform-specific BLAS calls inside `mixed_ld.rs`.

## 4. Rayon Row Work and Gram Construction

Residualization is independent by SNP row. Rayon partitions SNP rows in stable
indexed order to:

- subtract the retained fixed-effect projection;
- calculate each projected diagonal;
- mark finite/non-negligible rows;
- scale surviving residual rows to unit norm.

The final effective LD is computed as one symmetric Gram product of normalized
residuals. The returned matrix is explicitly symmetrized and its diagonal set
to exactly one to retain the existing contract.

The `valid_indices` result remains sorted by original SNP row. Parallel work
must not change filtering order, PIP row order, folded SNP restoration, or
credible-set membership.

## Error Handling

- Expected malformed BIM/genotype/data conditions continue to become
  `FineMapSkip` and preserve old paired outputs.
- Programming errors and BLAS contract violations remain fatal.
- If the requested BLAS backend is unavailable, use the repository's existing
  safe backend behavior; do not silently change statistical inputs.
- A kernel result must remain finite, symmetric, unit-diagonal, and positive
  semidefinite within the established roundoff tolerance.

## Testing

### BIM resolver

Tests use synthetic BIM files with:

- millions-scale irrelevant rows represented by a generated manageable
  benchmark fixture and an instrumented single-pass reader;
- duplicate coordinates with distinct IDs/alleles;
- missing SNP IDs and unambiguous coordinate fallback;
- mismatched IDs, ambiguous alleles, malformed rows, and duplicate resolution;
- a counting file wrapper proving one sequential pass and bounded retained
  candidates.

The legacy and streaming resolvers are compared on valid small fixtures before
the legacy path is removed.

### Numerical kernel

The current direct-`P`, rank-deficiency, filtering, thread-determinism,
synthetic FvLMM, and susieR parity tests remain mandatory. New tests compare
the BLAS/Rayon implementation with the pre-optimization scalar reference at:

- `atol=1e-10, rtol=1e-10` for effective LD on small deterministic arrays;
- `atol=1e-6, rtol=1e-5` for PIP and posterior means.

`threads=1,2,4,8` must preserve valid indices and numerical results.

### Performance

Use the same Rice6048 locus and `-mem 8`, with fresh output prefixes:

```text
jx postgwas ... -bimrange 3:18.414420:18.461203 -finemap susie \
  -finemap-max-iter 500 -finemap-L 5 -t {1,2,4,8}
```

Record per-stage wall time, total wall time, peak RSS, CPU utilization, backend,
SuSiE iterations, maximum posterior mean, PIP hash, and CS hash.

Acceptance criteria:

- BIM matching performs one sequential scan and no per-variant full scan.
- BIM retained Python object count scales with the regional candidates.
- `-t 2/4/8` is observable in EVD/GEMM diagnostics.
- Dense numerical stages show a useful trend where workload is large enough;
  no fixed speedup ratio is required for the 169-SNP locus because overhead
  may dominate.
- The optimized Rice6048 result remains converged at the same tolerance, with
  PIP/posterior mean within the numerical tolerances above.
- Peak RSS does not exceed the current memory preflight estimate or the
  requested `-mem` limit.

## Delivery and Repository Safety

Use a dedicated feature worktree and commits ordered by stage:

1. streaming BIM resolver;
2. timing/backend diagnostics;
3. BLAS matrix products;
4. Rayon residualization and BLAS Gram;
5. real-data benchmark evidence or test-only adjustments.

Generated Rice6048 results and benchmark logs stay under `/tmp` or ignored
local experiment paths and are never committed. No remote push is performed
without an explicit later request.
