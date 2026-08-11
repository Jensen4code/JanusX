# GWAS Log Null-Model Sidecar for Mixed-Model Fine-Mapping

## Purpose

JanusX currently computes SuSiE-RSS LD from raw genotype correlations. This is
appropriate for ordinary marginal models when the summary statistics and LD use
the same samples and allele coding, but it is not the null-statistic correlation
for a mixed model. FvLMM and SparseLMM summary statistics depend on the GRM,
variance components, fixed-effect projection, and exact GWAS sample order. Using
raw LD can therefore produce a summary-statistic/LD mismatch, non-convergence,
inflated prior variances, and implausibly large posterior effects.

The GWAS log will contain a versioned, machine-readable null-model sidecar for
each published result TSV. PostGWAS will match the input result to this metadata,
validate every dependency, and construct model-compatible LD. If validation
fails, fine-mapping will be skipped with a warning; it must never silently fall
back to raw LD for a mixed-model result.

## Scope

The first implementation supports FvLMM because all SNPs share one fitted null
variance ratio. The schema and matching layer are model-neutral so SparseLMM can
subsequently use the same contract with sparse solves.

The following are intentionally outside the first implementation:

- exact LMM fine-mapping when variance parameters are re-estimated per SNP;
- embedding full GRMs, phenotype vectors, sample ID lists, or covariate matrices
  in the log;
- searching outside the GWAS result directory for logs;
- silently using raw genotype LD after a sidecar lookup or validation failure.

Exact LMM input will be skipped with a warning until GWAS exposes summary
statistics calculated from one common null model.

## Log Format

The existing `<output-prefix>.gwas.log` remains human-readable. After a result
TSV has been finalized successfully, GWAS writes one JSON object between unique
sentinels to file handlers only:

```text
[JANUSX_GWAS_NULL_MODEL_V1_BEGIN]
{"schema":"janusx.gwas-null-model/v1", ...}
[JANUSX_GWAS_NULL_MODEL_V1_END]
```

Each block describes exactly one GWAS result. Multiple traits and models produce
multiple blocks in the same log. JSON is serialized deterministically with
sorted keys. Unknown fields are ignored by readers of the same major schema
version; an unknown schema major version is rejected.

Required fields for version 1 are:

- `schema`, `created_at`, and `janusx_version`;
- canonical `result_file`, result basename, byte size, and SHA-256;
- `model`, `trait`, output columns used to derive Z, and effective SNP count;
- genotype prefix and fingerprints for BED, BIM, and FAM;
- phenotype path and fingerprint, phenotype ID column, and trait column;
- covariate path/fingerprint and selected columns, or an explicit empty list;
- kinship path, format, matrix shape, ID-file path, and fingerprints;
- sample count and SHA-256 of the ordered sample ID sequence;
- null-model `lambda`, `sigma_g2`, `sigma_e2`, PVE, GRM trace mean, and fixed
  effect column descriptions;
- genotype filters and allele-coding convention needed to reproduce the scan.

Paths are recorded canonically for exact local matching. Fingerprints allow a
result and its dependencies to be verified after files are moved together. Full
sample IDs and numerical matrices are not embedded, keeping the log bounded for
biobank-scale datasets.

## GWAS Publication Semantics

The sidecar block is emitted only after the corresponding result TSV has been
atomically finalized. Metadata is built from the fitted model and the aligned
objects actually used by the scan, rather than reconstructed from CLI arguments.

GWAS computes the ordered-sample hash from an unambiguous length-prefixed byte
encoding of sample IDs. File fingerprints use streamed SHA-256 and do not load
large files into memory. If sidecar serialization or logging fails, the GWAS TSV
remains valid, but GWAS emits a warning that mixed-model fine-mapping metadata is
unavailable for that result.

The structured JSON block is written only to the `.gwas.log` file. The terminal
receives one concise message confirming that null-model metadata was recorded.

## PostGWAS Discovery and Matching

For each `-i` result passed to `-finemap susie`, PostGWAS searches only
`*.gwas.log` files in the result TSV directory. It parses complete sentinel
blocks and ignores ordinary log text.

Candidate matching proceeds in this order:

1. canonical result path;
2. result basename, byte size, and SHA-256 for a moved result;
3. model and trait as consistency checks, never as identity by themselves.

Exactly one candidate must remain. No candidate, multiple candidates, malformed
JSON, or an unsupported schema produces a warning and skips fine-mapping.

After matching, PostGWAS verifies the result and all referenced inputs. It
reconstructs the GWAS sample set and order from FAM, phenotype, missingness, and
covariate metadata, then checks the ordered-sample hash. It also verifies GRM ID
order, matrix dimensions, file fingerprints, model, trait, allele convention,
and genotype prefix compatibility with the user-supplied `-bfile`.

Any mismatch produces a warning with the failed field and skips all requested
fine-mapping loci for that result. PostGWAS does not publish new PIP or credible-
set files and explicitly warns that pre-existing files with the same output
prefix may belong to an earlier run.

## FvLMM Effective LD

For a fitted FvLMM null model,

\[
V \propto K + \lambda I.
\]

Let `C` be the exact fixed-effect design used by GWAS, including the intercept,
and let `G` contain locus genotypes in the verified GWAS sample order. PostGWAS
constructs

\[
P = V^{-1} - V^{-1}C(C^\mathsf{T}V^{-1}C)^{-1}C^\mathsf{T}V^{-1},
\]

\[
Q = G^\mathsf{T}PG,
\qquad
R_{\mathrm{FvLMM}} = D^{-1/2}QD^{-1/2},
\qquad
D = \operatorname{diag}(Q).
\]

The implementation should use the GRM eigensystem or linear solves and must not
materialize `P` as a dense matrix. Fixed effects are projected on the weighted
scale. It must not explicitly invert `C^T V^-1 C`: user covariates can be exactly
or nearly collinear after adding the intercept. The weighted fixed-effect system
is solved with a rank-revealing SVD/pseudoinverse or an equivalently robust
rank-revealing QR method using an explicit relative singular-value tolerance.
PostGWAS records the detected fixed-effect rank and tolerance. Non-full rank is
accepted when the projected result is finite and stable; an unusable rank-zero,
non-finite, or numerically unresolved system warns and skips fine-mapping.

Monomorphic or numerically degenerate variants are removed before normalization,
and the resulting matrix is symmetrized with a unit diagonal.

The existing `-mem` PostGWAS limit applies to genotype blocks, GRM/eigensystem
workspace, effective-LD construction, and SuSiE. A memory preflight occurs before
large allocations.

The first implementation uses the FvLMM Wald Z already present in the result
TSV, while recording that the LD is based on the shared null model. A later
extension may output common-null score Z directly; exact LMM remains unsupported
until that statistic is available.

## Clumping and SuSiE

Fine-mapping clumping operates on the same verified GWAS samples. Its default
threshold remains a separate policy decision and is not treated as a substitute
for mixed-model effective LD.

After clumping, SuSiE receives the aligned Z vector and effective LD matrix.
Existing allele checks, folded-SNP restoration, credible-set construction,
temporary-file publication, backup, and rollback behavior remain in force.

PostGWAS records the following diagnostics in its log:

- matched GWAS log and sidecar schema;
- null-model type, lambda, PVE, and verified sample count;
- effective-LD minimum eigenvalue and condition number;
- fixed-effect column count, effective rank, and rank tolerance;
- SuSiE convergence state, iteration count, and prior variances;
- the reason for every warning-triggered skip.

## Error and Warning Policy

The following conditions warn and skip fine-mapping without raising a fatal
PostGWAS process error:

- missing, ambiguous, malformed, or unsupported sidecar metadata;
- result or dependency fingerprint mismatch;
- sample-set/order or GRM-ID mismatch;
- missing null-model parameters;
- unsupported mixed-model method;
- memory preflight failure specific to fine-mapping;
- effective LD that cannot be normalized safely.

The process may continue with unrelated plotting, annotation, other input files,
or other loci/results whose metadata is valid. No raw-LD fallback is permitted
for an identified mixed-model input.

Internal programming errors and corrupt output-publication state remain fatal so
they are not hidden as ordinary compatibility warnings.

## Compatibility

Existing GWAS TSV columns and ordinary log text remain unchanged. Older logs do
not contain a sidecar; mixed-model fine-mapping against those results warns and
skips. LM fine-mapping continues to use sample-matched ordinary LD and does not
require mixed-model metadata.

PostGWAS may expose an advanced/development-only explicit GWAS-log override for
moved or nonstandard layouts. Automatic same-directory discovery remains the
normal interface, so `-finemap susie` is the only new production-facing option.

## Testing

Tests will cover:

1. deterministic sidecar serialization and multiple blocks per log;
2. exact-path and moved-file fingerprint matching;
3. missing, ambiguous, malformed, and unsupported sidecars;
4. dependency, sample-order, and GRM-ID mismatches;
5. warning-and-skip behavior with no newly published PIP/CS files;
6. FvLMM effective LD against a direct small-matrix reference implementation;
7. equivalence of effective LD from eigensystem and solve formulations;
8. exact and near rank-deficient covariates, compared with a direct
   Moore-Penrose reference projection and checked for finite deterministic LD;
9. a synthetic mixed-model locus where raw LD is unstable but effective LD
   converges without posterior-effect inflation;
10. the Rice6048 FvLMM regression case, checking null-model reconstruction,
   convergence, bounded prior variance, and agreement with susieR;
11. unchanged LM fine-mapping and unrelated GWAS output behavior.

Large generated datasets and experiment outputs remain local test artifacts and
are not committed or pushed.
