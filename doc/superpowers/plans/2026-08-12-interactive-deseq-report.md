# Interactive DESeq2 Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an offline-capable interactive HTML report for both DESeq2 sample versions, with clickable UpSet intersections, annotated gene tables, and interactive volcano plots.

**Architecture:** A Python standard-library generator will read the existing JXFU result tree, use the installed `plotly` package only to obtain the self-contained Plotly JavaScript bundle, and write an HTML application plus local JavaScript data files. The HTML loads one set payload and one volcano payload on demand, so the browser does not hold all 104 comparisons at startup. The generator will be run on JXFU through `dsub` in the `jxfu` micromamba environment; it will not rerun alignment, counting, or DESeq2.

**Tech Stack:** Python 3, `csv`/`json`/`html`/`math`/`pathlib`, Plotly JavaScript embedded by `plotly.offline.get_plotlyjs()`, browser-native JavaScript/CSS/HTML, JXFU `dsub` and `djob`.

## Global Constraints

- Input root: `/share/home/jxfu26/03_counts_star/deseq_upset_20260812/`.
- Output root: `/share/home/jxfu26/03_counts_star/deseq_upset_20260812/html_report/`.
- Sample branches: `all_samples` and `remove_R-M-1d-3_S-M-1d-3`.
- Model directory mapping: `main` -> `main_effects`; `strict` -> `strict_matched`.
- UpSet directory mapping: `main` -> `upset_plots/main`; `strict` -> `upset_plots/strict`.
- Each branch has 52 comparisons: 6 main and 46 strict; the report has 104 comparison volcano payloads.
- Difference thresholds remain `padj < 0.05` and `abs(log2FoldChange) > 1`.
- Gene tables show `2^log2FoldChange` as linear FC but use the original DESeq2 values for filtering.
- The report must not require a CDN or external network resource.
- HPC work must be submitted with `dsub`; use `djob` for monitoring and verification.
- Do not overwrite an existing report unless the generator receives an explicit `--force` flag.
- Preserve the unrelated untracked PCA/clustering documents already present in the repository.

---

### Task 1: Add testable input and payload builders

**Files:**
- Create: `scripts/rnaseq_interactive_report.py`
- Create: `scripts/tests/test_rnaseq_interactive_report.py`

**Interfaces:**
- `load_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]` reads UTF-8 tab-separated files and preserves empty fields.
- `load_comparison_manifest(input_root: Path) -> list[dict[str, str]]` reads the top-level manifest and returns rows in file order.
- `model_dir(model: str) -> str` returns `main_effects` for `main` and `strict_matched` for `strict`, raising `ValueError` for any other model.
- `payload_slug(branch: str, comparison_id: str) -> str` returns a collision-free filename stem using the branch and comparison ID.
- `load_annotation_reference(input_root: Path) -> dict[str, dict[str, str]]` returns one annotation row keyed by `Geneid` and raises on duplicate keys.
- `build_set_payload(input_root: Path, branch: str, model: str, category: str, direction: str, manifest_rows: list[dict[str, str]], annotations: dict[str, dict[str, str]]) -> dict` returns a JSON-serializable payload with `sets`, `intersections`, and result `records`.
- `build_volcano_payload(input_root: Path, branch: str, comparison: dict[str, str], annotations: dict[str, dict[str, str]]) -> dict` returns compact background points plus fully annotated significant points.

- [ ] **Step 1: Write failing unit tests for path mapping and TSV parsing**

Create a temporary fixture containing one `main` result, one annotated gene-set file, one `gene_membership.tsv`, one `intersection_table.tsv`, one `set_manifest.tsv`, a branch manifest, and a two-row annotation reference. Assert:

```python
def test_model_directory_mapping():
    self.assertEqual(model_dir("main"), "main_effects")
    self.assertEqual(model_dir("strict"), "strict_matched")
    with self.assertRaises(ValueError):
        model_dir("main_effects")

def test_load_tsv_preserves_empty_fields(self):
    header, rows = load_tsv(self.fixture / "empty.tsv")
    self.assertEqual(header, ["Geneid", "Description"])
    self.assertEqual(rows[0]["Description"], "")
```

Run:

```bash
python -m unittest scripts.tests.test_rnaseq_interactive_report -v
```

Expected: FAIL because the module and functions do not yet exist.

- [ ] **Step 2: Implement the parser and path mapping**

Use `csv.DictReader(..., delimiter="\t", lineterminator="\n")`, validate that every row has a non-empty `Geneid` where a gene key is required, and implement the two explicit model mappings. Do not use shell parsing or pandas so the generator remains runnable in the JXFU environment with only Python’s standard library plus Plotly for the JavaScript bundle.

- [ ] **Step 3: Add failing payload tests**

Extend the fixture with two significant genes and assert the exact payload contract:

```python
def test_set_payload_contains_intersection_records(self):
    payload = build_set_payload(
        self.root, "all_samples", "main", "Disease", "Up",
        manifest_rows, annotations,
    )
    self.assertEqual(payload["sets"], ["S_vs_R"])
    self.assertEqual(payload["intersections"][0]["count"], 1)
    self.assertEqual(payload["intersections"][0]["sets"], ["S_vs_R"])
    self.assertEqual(payload["intersections"][0]["record_keys"], ["main_Disease_S_R|Zm00001"])
    self.assertEqual(payload["records"]["main_Disease_S_R|Zm00001"]["linearFC"], 4.0)

def test_volcano_payload_keeps_all_background_and_annotations(self):
    payload = build_volcano_payload(self.root, "all_samples", comparison, annotations)
    self.assertEqual(len(payload["background"]), 2)
    self.assertEqual(payload["significant"][0]["Geneid"], "Zm00001")
    self.assertEqual(payload["significant"][0]["eggNOG_GOs"], "GO:0001")
```

Run the focused tests and confirm they fail for the missing payload builders.

- [ ] **Step 4: Implement set payload construction**

Read the actual paths:

```text
{branch}/{main_effects|strict_matched}/gene_sets/{Category}/{ComparisonID}_{Up|Down}.tsv
{branch}/upset_plots/{main|strict}/{Category}/{Up|Down}/gene_membership.tsv
{branch}/upset_plots/{main|strict}/{Category}/{Up|Down}/intersection_table.tsv
{branch}/upset_plots/{main|strict}/{Category}/{Up|Down}/set_manifest.tsv
```

Build `records` keyed by `ComparisonID|Geneid`. Add a numeric `linearFC` field as `2.0 ** log2FoldChange`, using `null` for a non-finite value. Build each intersection’s `record_keys` by matching the boolean membership pattern in `gene_membership.tsv`, and retain `count`, `sets`, and `code` from `intersection_table.tsv`. Preserve all annotated gene-set columns in each record. Sort intersections by descending count and then stable intersection ID.

- [ ] **Step 5: Implement volcano payload construction**

Read the result table from `{branch}/{model_dir(model)}/results/{ComparisonID}.tsv`. Store every finite `log2FoldChange` row in `background` as a compact array `[Geneid, x, y, padj]`, where `y = -log10(max(padj, floor))`; use a floor based on the smallest positive finite padj in that result and `1e-300` as the lower bound. Store significant rows in `significant` as full objects with the result statistics, `linearFC`, and every annotation column. Include `thresholds`, `comparison`, and the original result path in the payload metadata. Keep raw `padj` unchanged in the significant object even when the plotted y value is capped.

- [ ] **Step 6: Run the unit tests**

Run:

```bash
python -m unittest scripts.tests.test_rnaseq_interactive_report -v
python -m py_compile scripts/rnaseq_interactive_report.py
```

Expected: all parser and payload tests PASS and `py_compile` exits with code 0.

- [ ] **Step 7: Commit the data-builder implementation**

```bash
git add scripts/rnaseq_interactive_report.py scripts/tests/test_rnaseq_interactive_report.py
git commit -m "feat: add interactive DESeq2 report data builders"
```

---

### Task 2: Add report materialization and self-contained HTML application

**Files:**
- Modify: `scripts/rnaseq_interactive_report.py`
- Create: `scripts/rnaseq_interactive_report_template.html`
- Modify: `scripts/tests/test_rnaseq_interactive_report.py`

**Interfaces:**
- `write_js_payload(path: Path, variable: str, payload: dict) -> None` writes a browser-loadable JavaScript assignment using JSON encoding.
- `generate_report(input_root: Path, output_dir: Path, force: bool = False) -> dict[str, int]` writes the report atomically and returns counts for branches, comparisons, set payloads, and volcano payloads.
- CLI: `python scripts/rnaseq_interactive_report.py --input-root INPUT --output-dir OUTPUT [--force]`.

- [ ] **Step 1: Write failing output-contract tests**

Add a test that calls `generate_report` on the small fixture and asserts:

```python
summary = generate_report(self.root, self.output)
self.assertEqual(summary, {"branches": 1, "comparisons": 1, "set_payloads": 1, "volcano_payloads": 1})
self.assertTrue((self.output / "index.html").is_file())
self.assertTrue((self.output / "data/manifest.js").is_file())
self.assertTrue((self.output / "data/sets/all_samples__main__Disease__Up.js").is_file())
self.assertTrue((self.output / "data/volcano/all_samples__main_Disease_S_R.js").is_file())
self.assertNotIn("https://", (self.output / "index.html").read_text())
```

Expected: FAIL because the writer and template do not yet exist.

- [ ] **Step 2: Implement output staging and JavaScript writers**

Create a sibling staging directory named `.html_report.partial`, refuse to use it if a previous partial directory exists, and write all files under it. Create `data/manifest.js`, 24 set payload files, and 104 volcano payload files. Name volcano files `data/volcano/{branch}__{ComparisonID}.js` because the two sample branches intentionally reuse the same `ComparisonID` values. Include in the manifest the branch names, sample counts read from each `metadata_used.tsv`, model/category lists, comparison metadata, thresholds, payload paths, and generator version. Replace an existing output only when `--force` is supplied; use a directory rename after all files and the completion marker are written.

The payload assignment format must be:

```javascript
window.__RNASEQ_REPORT_MANIFEST = {...};
window.__RNASEQ_SET_PAYLOAD = {...};
window.__RNASEQ_VOLCANO_PAYLOAD = {...};
```

Each lazy-loaded data file must set only its corresponding `window` variable and the HTML loader must delete the variable after consuming it.

- [ ] **Step 3: Implement the offline HTML template**

The template must contain the Plotly bundle from `plotly.offline.get_plotlyjs()` inline and load only `data/manifest.js` plus dynamically selected local data files. It must not reference a CDN, a remote font, or an external stylesheet. Provide these DOM elements with stable IDs:

```text
branchSelect, modelSelect, categorySelect, comparisonSelect, directionSelect
comparisonSummary, upsetBars, upsetMatrix, intersectionSelect
geneSearch, geneTable, geneTableStatus, geneAnnotation
volcanoPlot, volcanoAnnotation, volcanoStatus
```

Use CSS grid to keep selectors and summary at the top, UpSet panels above the gene table, and the volcano plot beside its annotation panel on wide screens; collapse to one column below 1100px.

- [ ] **Step 4: Implement selector and lazy-loader behavior**

On startup, load the manifest and populate branch/model/category/comparison/direction selectors. When a selector changes, derive the set payload key and comparison payload path from the manifest. Load a local JavaScript file with a unique `<script>` element, wait for its `onload`, consume the corresponding window variable, then remove the script element and variable. Show a visible error card with the failing relative path if the script fails.

- [ ] **Step 5: Implement clickable UpSet**

Render the top 80 intersections by count in a Plotly bar chart and render the same intersection IDs in a dot-matrix chart. Keep all intersections in `intersectionSelect` so a user can choose an intersection that is outside the displayed top 80. Clicking a bar or matrix column selects the same intersection and calls `renderGeneTable(record_keys)`. The default selection is the first non-empty intersection after descending count sort. For empty Up/Down data, replace both plots with an empty-state message.

- [ ] **Step 6: Implement sortable/searchable annotated gene table**

Render 50 rows per page from the selected intersection. Default sort is `abs(log2FoldChange)` descending; clicking a header toggles ascending/descending order. Search must match `Geneid`, `Description`, `Preferred_name`, GO, KEGG KO, or KEGG pathway case-insensitively. Show the columns `Geneid`, `ComparisonID`, `Sets`, `Chr`, `Start`, `End`, `padj`, `log2FoldChange`, `linearFC`, `eggNOG_Description`, `eggNOG_Preferred_name`, `eggNOG_GOs`, and `eggNOG_KEGG_ko`, with a “more annotations” expansion for EC, pathway, module, reaction, COG, PFAM, and OG fields. Clicking a gene row renders all available fields in `geneAnnotation`, sets the volcano highlight target, and changes `geneSearch` to the selected gene.

- [ ] **Step 7: Implement interactive volcano plot and annotation panel**

Load the selected branch-qualified volcano payload. Render `background` as a low-opacity gray Plotly scatter layer and `significant` as separate Up/Down scatter layers. Add threshold lines at `x=-1`, `x=1`, and `y=-log10(0.05)`. Use `customdata` on significant points to carry the full row object; its hover template must include `Geneid`, `Chr`, `Start`, `End`, `padj`, `log2FoldChange`, `linearFC`, description, Preferred name, GO, KEGG KO, and KEGG pathway. Background hover shows `Geneid`, `log2FoldChange`, and raw `padj`.

On `plotly_click` for a significant point, render the complete annotation panel and, if the gene exists in the currently loaded set payload, select its intersection and scroll to the table; otherwise set the table search to the gene and show “当前交集不包含该基因，可切换到当前方向全部显著基因查看”。 Use `Plotly.restyle` or a dedicated highlight trace to mark the selected point without adding a five-point star.

- [ ] **Step 8: Add output-contract tests and run them**

Extend the fixture test to parse every generated JavaScript file with a small extraction helper, verify the manifest counts, verify one set record’s `linearFC`, verify one volcano point’s GO/KEGG fields, and scan `index.html` for `http://`, `https://`, and `//cdn`. Run:

```bash
python -m unittest scripts.tests.test_rnaseq_interactive_report -v
python -m py_compile scripts/rnaseq_interactive_report.py
```

Expected: all tests PASS and no external resource marker is found.

- [ ] **Step 9: Commit the HTML generator**

```bash
git add scripts/rnaseq_interactive_report.py scripts/rnaseq_interactive_report_template.html scripts/tests/test_rnaseq_interactive_report.py
git commit -m "feat: generate interactive DESeq2 HTML report"
```

---

### Task 3: Package and submit the full report build on JXFU

**Files:**
- No repository files beyond the Task 1 and Task 2 commits.
- Remote staging: `/share/home/jxfu26/tmp_deseq_upset_html_report_20260812/`.
- Remote final output: `/share/home/jxfu26/03_counts_star/deseq_upset_20260812/html_report/`.

**Interfaces:**
- The remote job consumes the committed generator and existing result root.
- The remote job produces the final report directory and a job log.

- [ ] **Step 1: Verify the remote environment and queue**

Run from a login shell:

```bash
ssh -p 18083 jxfu26@211.69.141.180 "bash -l -c 'command -v dsub; command -v djob; dqueue; /share/home/jxfu26/software/bin/micromamba run -n jxfu python -c \"import plotly, sys; from plotly.offline import get_plotlyjs; print(sys.executable); print(plotly.__version__); print(len(get_plotlyjs()))\"'"
```

Record the available queue and the Plotly bundle check before submission. Do not use `bsub` or `csub` for this task.

- [ ] **Step 2: Upload only the committed generator files**

Create the remote staging directory, then copy the two generator files and the unit-test file with `scp` over port `18083`. Confirm their SHA-256 hashes on both sides before submitting. This transfer is file staging; the compute task itself remains a `dsub` job.

- [ ] **Step 3: Submit one full report job with dsub**

Use a queue returned by `dqueue` and request eight CPUs and 16 GB memory. The command shape is:

```bash
dsub -q <queue-from-dqueue> -R "cpu=8,mem=16GB" \
  --cwd /share/home/jxfu26/tmp_deseq_upset_html_report_20260812 \
  -o /share/home/jxfu26/tmp_deseq_upset_html_report_20260812/report.out \
  -e /share/home/jxfu26/tmp_deseq_upset_html_report_20260812/report.err \
  /share/home/jxfu26/software/bin/micromamba run -n jxfu python \
  /share/home/jxfu26/tmp_deseq_upset_html_report_20260812/rnaseq_interactive_report.py \
  --input-root /share/home/jxfu26/03_counts_star/deseq_upset_20260812 \
  --output-dir /share/home/jxfu26/03_counts_star/deseq_upset_20260812/html_report
```

Record the returned job ID, queue, requested resources, and submission time. The generator must fail instead of replacing a pre-existing report unless `--force` is intentionally added after checking its contents.

- [ ] **Step 4: Monitor the existing job**

Poll the returned job ID with:

```bash
ssh -p 18083 jxfu26@211.69.141.180 "bash -l -c 'djob JOB_ID -l; djob JOB_ID --step'"
```

Do not resubmit while the original job is pending or running. If it fails, inspect `report.err` and the job detail before changing the command.

---

### Task 4: Validate the generated report and browser-facing behavior

**Files:**
- Remote: `/share/home/jxfu26/03_counts_star/deseq_upset_20260812/html_report/`.
- Create locally: `scripts/validate_rnaseq_interactive_report.py`.

**Interfaces:**
- `validate_report(report_dir: Path, input_root: Path) -> dict[str, int]` validates counts, paths, payload keys, and selected value equality against source TSV files.

- [ ] **Step 1: Add validation tests for the validator**

Use the Task 2 fixture to test a valid report and then delete one volcano payload and assert that validation raises an error naming the missing comparison ID.

- [ ] **Step 2: Implement structural and source-equality validation**

The validator must check:

```text
manifest branches = 2
manifest comparisons = 104
set payload files = 24
volcano payload files = 104
every manifest payload path exists and has non-zero size
every result ComparisonID occurs once per branch
all significant record Geneid values are in the annotation reference or explicitly have empty annotation fields
all set record values for a sampled gene match its annotated gene-set TSV
all sampled volcano x/padj values match its result TSV
index.html contains the inline Plotly bundle and no URL beginning with http:// or https://
```

Use deterministic samples: full/all-samples `main_Disease_S_R`, reduced `main_Day_3d_0d`, and full/strict `strict_Disease_S_L_0d_R_L_0d_S_R`. Report sizes and counts in JSON and TSV validation summaries.

- [ ] **Step 3: Run validation on JXFU after the job finishes**

Run:

```bash
ssh -p 18083 jxfu26@211.69.141.180 "bash -l -c '/share/home/jxfu26/software/bin/micromamba run -n jxfu python /share/home/jxfu26/tmp_deseq_upset_html_report_20260812/validate_rnaseq_interactive_report.py --input-root /share/home/jxfu26/03_counts_star/deseq_upset_20260812 --report-dir /share/home/jxfu26/03_counts_star/deseq_upset_20260812/html_report'"
```

Expected: exit code 0, 104 branch-qualified volcano payloads, 24 set payloads, and all deterministic source-equality checks pass.

- [ ] **Step 4: Verify offline opening and interaction hooks**

Use a local HTTP server against a copied report directory or a browser that permits local scripts. Verify the following visible actions:

1. changing `all_samples` to the reduced branch changes the summary and comparison list;
2. clicking an UpSet bar changes the gene table;
3. sorting by `linearFC` changes row order;
4. searching a GO or KEGG term filters the table;
5. hovering a significant volcano point shows its gene and function fields;
6. clicking that point fills the annotation panel and updates the gene filter;
7. selecting an empty direction displays the empty-state message without a JavaScript error.

- [ ] **Step 5: Record the final handoff evidence**

Collect `djob JOB_ID -ll`, output `du -sh`, report file counts, validation JSON, and SHA-256 hashes of `index.html`, `data/manifest.js`, and the generator. Include the exact JXFU path and local opening command in `README.md` generated inside the report.

- [ ] **Step 6: Commit the validator**

```bash
git add scripts/validate_rnaseq_interactive_report.py scripts/tests/test_rnaseq_interactive_report.py
git commit -m "test: validate interactive DESeq2 report artifacts"
```

## Self-review checklist

- The plan maps the approved design’s two sample branches, both models, six main comparisons, 46 strict comparisons, UpSet interactions, annotated gene tables, volcano hover/click behavior, lazy loading, offline resources, and source-equality validation to explicit tasks.
- The actual existing paths use `upset_plots/main` and `upset_plots/strict`, while annotated gene sets use `main_effects` and `strict_matched`; the plan keeps these mappings separate.
- No step changes the DESeq2 thresholds or statistical model.
- No placeholder, external CDN, `bsub`, or `csub` command is used.
- The report generator is tested against a small fixture before a full remote build.
