# kformat Windowed Input I/O Design

## Goal

Make `jx kformat` memory-bounded on Linux HPC cgroups for very large packed
k-mer matrices, without changing row filtering, output ordering, or the
existing `-t/--thread` behavior.

## Evidence and constraints

The current implementation bounds Rust-owned selected-row vectors but maps the
entire `.bkmer` and `.bsite` files.  On JXFU26, file-backed pages touched by
those mappings are charged to the job cgroup; a 119 GiB `.bkmer` plus a 476 GiB
`.bsite` reached the scheduler memory limit and was killed.

The replacement must therefore avoid whole-file mappings during conversion.
It must also avoid loading an extract list or a selected-row index proportional
to the number of input k-mers.  The existing output formats and metadata remain
unchanged.

## Architecture

`kformat` will validate only the small binary headers and file lengths during
layout loading.  Conversion will open the two input files as ordinary files
and process fixed row blocks with positional reads (`read_exact_at`).

Each worker receives a block descriptor, reads the corresponding `.bkmer` and
`.bsite` byte ranges into block-local buffers, computes the selected local row
offsets, and returns the buffers plus offsets.  The ordered consumer retains a
bounded number of completed blocks in flight and invokes the existing writer
callback for selected rows in source order.  Writer callbacks receive the
current row number, k-mer code, and presence bytes from the current block rather
than borrowing from a global mmap.

The block size remains 16,384 rows.  The in-flight window is bounded by the
worker/channel capacity, so peak input buffers are proportional to worker count
and sample count, not total k-mer count.  With 249 samples, one full block is
about 0.625 MiB before the selected-offset vector.

After an ordered block has been consumed, Unix builds will issue a best-effort
`posix_fadvise(POSIX_FADV_DONTNEED)` for its input ranges.  This prevents the
kernel page cache from accumulating the sequential scan under a cgroup.  The
conversion remains correct if the advisory call is unavailable or fails.

## Error handling

- Header, metadata, and file-length mismatches fail before output conversion.
- Positional reads use checked offset/length arithmetic and fail on short reads.
- A worker read or filter error terminates the bounded pipeline and removes the
  temporary output directory through the existing cleanup path.
- Advisory page-cache calls are non-fatal diagnostics/optimizations.

## Compatibility

The Python CLI signature is unchanged.  `-t/--thread`, MAF filtering, extract
filtering, bfile output, kfile output, temporary-output commit behavior, and
deterministic row order retain their existing semantics.

## Testing

Unit tests will cover positional block reads, truncated input, single-thread
and multi-thread ordering across block boundaries, kfile header patching, and
bfile row conversion.  Existing kformat tests and the Python smoke test must
continue to pass.  A local memory smoke test will verify that the reader holds
only a bounded number of block buffers; the real 595 GiB HPC input will be
validated separately with `/usr/bin/time -v` and scheduler peak-memory data.

