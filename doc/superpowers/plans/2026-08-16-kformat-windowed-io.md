# kformat Windowed Input I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Replace `kformat` whole-file mmap conversion with bounded positional block reads so HPC cgroup memory stays independent of total k-mer count.

**Architecture:** Validate binary headers and exact file lengths with ordinary file reads. A shared `KformatInput` owns the two input files and reads 16,384-row blocks with checked `read_exact_at` offsets. Bounded worker threads return block-local input buffers and selected offsets; the ordered consumer writes rows from those buffers and issues best-effort Linux `POSIX_FADV_DONTNEED` advice after each consumed block.

**Tech Stack:** Rust 2021, PyO3 extension, `std::os::unix::fs::FileExt::read_exact_at`, `libc::posix_fadvise`, existing BufWriter and kfile/bfile formats.

## Global Constraints

- Do not map whole `.bkmer` or `.bsite` files during conversion.
- Keep `-t/--thread`, MAF filtering, extract filtering, output order, and output bytes compatible.
- Keep in-flight input blocks bounded by the existing worker/channel window.
- Treat page-cache advice as best effort; correctness cannot depend on it.
- Do not add dependencies or stage files under `test/` or benchmark directories.

---

### Task 1: Add failing window-reader and pipeline tests

**Files:**
- Modify: `src/kmer/kformat.rs` test module
- Test: `src/kmer/kformat.rs` unit tests

**Interfaces:**
- Tests will exercise `KformatInput::open`, `KformatInput::read_block`, and the refactored `stream_selected_rows` callback contract.

- [ ] **Step 1: Add a test fixture helper that writes valid packed input files.**

Create a test-only helper that writes the real binary headers followed by supplied `.bkmer` and `.bsite` bodies into a unique temporary directory. It must return a `KformatLayout` whose paths point to those files and remove the directory at the end of each test.

- [ ] **Step 2: Add a failing positional-read test.**

Write a fixture with more than one `DEFAULT_SCAN_BLOCK_ROWS` block, open it through the wished-for `KformatInput::open`, read a middle block, and assert that its first and last row bytes match the source body. Run:

```bash
cargo test --offline kmer::kformat::tests::window_reader_reads_middle_block --lib
```

Expected result before implementation: compile failure because `KformatInput` is not defined.

- [ ] **Step 3: Add a failing short-read test.**

Truncate one fixture payload after writing a valid header and assert that `KformatInput::open` or `read_block` returns an error containing `payload length` or `short read`. Run the focused test and confirm the failure is due to the missing reader behavior, not a test typo.

- [ ] **Step 4: Update the selection tests to the block-reader API.**

Change the existing source-order, count, and multi-block ordering tests to create a fixture and call the new `stream_selected_rows` callback with `(row, code, kmer_bytes, presence_bytes)`. Keep assertions on selected row order and payload bytes.

### Task 2: Implement bounded positional input and block filtering

**Files:**
- Modify: `src/kmer/kformat.rs`

**Interfaces:**
- Add `KformatInput::open(layout: &KformatLayout) -> Result<Self>`.
- Add `KformatInput::read_block(block_idx: usize, start: usize, end: usize) -> Result<InputBlock>`.
- Add `InputBlock { block_idx, start, bkmer, bsite, selected }`.
- Refactor `stream_selected_rows` to accept `&KformatInput` and call `FnMut(usize, u64, &[u8], &[u8])` for each selected source row.

- [ ] **Step 1: Add header-only validation.**

Replace `load_layout`’s full-file `open_*_mmap` validation with ordinary file opens, fixed-size header reads, `parse_bkmer_header`/`parse_bsite_header`, and `fs::metadata` length checks. Preserve every existing metadata mismatch error and close the validation handles before conversion.

- [ ] **Step 2: Implement checked positional reads.**

Use `FileExt::read_exact_at` on Unix and a cloned-file seek/read fallback for non-Unix builds. Calculate header-plus-row offsets with checked `u64` arithmetic. Allocate exactly `8 * rows` and `bytes_per_col * rows` bytes for each block and return an error on overflow or short read.

- [ ] **Step 3: Refactor block-local filtering.**

Make `row_is_selected` and `selected_rows_in_block` consume block-local slices and local row indices. The worker reads its input block, filters it, and returns the block buffers plus `Vec<u32>` offsets. Preserve extract short-circuiting and MAF semantics.

- [ ] **Step 4: Refactor the bounded ordered pipeline.**

Keep the existing worker count, sync-channel capacity, dispatch window, and `BTreeMap` ordering logic. Replace global mmap slices with `KformatInput::read_block`; when consuming a block, derive the code and presence slice from that block and invoke the new callback. Drop the block immediately after its rows are written.

- [ ] **Step 5: Run the Task 1 focused tests.**

Run:

```bash
cargo test --offline kmer::kformat::tests --lib
```

Expected result: all kformat tests pass, including the new positional-read and short-read tests.

### Task 3: Update bfile/kfile writers and page-cache release

**Files:**
- Modify: `src/kmer/kformat.rs`

**Interfaces:**
- `write_bfile` and `write_kfile` consume `&KformatInput` rather than body slices.
- `run_kformat` opens one `KformatInput` and no longer calls `open_*_mmap`, `Advice::Sequential`, or full-body accessors.

- [ ] **Step 1: Adapt bfile writing.**

Use the callback’s decoded code and block-local presence bytes for `.bim` and `.bed` output. Keep PLINK row encoding and output counters unchanged.

- [ ] **Step 2: Adapt kfile writing.**

Write the callback’s original eight k-mer bytes and presence bytes, preserving exact payload order. Keep placeholder headers, seek-back count patching, metadata, and temporary-output commit behavior unchanged.

- [ ] **Step 3: Add Linux page-cache discard advice.**

After ordered consumption of each block, call `libc::posix_fadvise` with `POSIX_FADV_DONTNEED` for the corresponding input ranges on Linux. Ignore non-zero advisory return codes and compile the helper as a no-op on platforms without this API.

- [ ] **Step 4: Run formatting and focused tests.**

Run:

```bash
cargo fmt --all -- --check
git diff --check
cargo test --offline kmer::kformat::tests --lib
```

### Task 4: Validate extension behavior and bounded-memory smoke test

**Files:**
- Modify: none in tracked test/benchmark directories
- Use: ignored local smoke script under `test/` or a temporary script outside the repository

**Interfaces:**
- Python `jxrs.kformat_run` remains unchanged.

- [ ] **Step 1: Rebuild the local PyO3 extension.**

Use the `jxfu` environment and the existing `maturin develop --release --locked --features python-extension` workflow from the JanusX project instructions.

- [ ] **Step 2: Run kfile and bfile Python smoke conversions.**

Verify both output formats, MAF filtering, multi-thread ordering, and header counts with a small generated fixture.

- [ ] **Step 3: Run a larger local memory smoke test.**

Generate enough rows to span many blocks and execute under `/usr/bin/time -l`; verify RSS stays near a fixed bound as row count increases rather than scaling with total input bytes.

- [ ] **Step 4: Review the final diff and record known unrelated failures.**

Inspect staged paths to ensure no local test/benchmark script is committed. Run the focused suite and report the existing unrelated full-suite failures separately.

- [ ] **Step 5: Commit implementation.**

```bash
git add src/kmer/kformat.rs
git diff --cached --name-only
git commit -m "fix(kformat): bound input memory with windowed reads"
```

