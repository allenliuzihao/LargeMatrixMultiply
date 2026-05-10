# LargeMatrixMultiply

This is a C++ library project implementing matrix and vector primitives and unit tests with sparse matrix and vector optimizations techniques including sparse vector representation and sparse matrix representation using CSR (row-based) and CSC (column-based) Compression Sparse Formats.

## Project summary

This repository contains native C++ implementations for matrix and vector operations used for large-scale numerical work. The code is organized as a Visual Studio project and includes a small unit-test project under `Tests/`.

Key source files:

- `Matrix.cpp`, `Matrix.h` — matrix operations and helpers
- `Vector.cpp`, `Vector.h` — vector utilities
- `Utilities.cpp`, `Utilities.h` — supporting utilities and helpers
- `Tests/` — unit tests (GoogleTest or custom test harness via Visual Studio Test project)

## Requirements

- Windows 10 or later
- Visual Studio 2026 with C++ workload (MSVC)
- C++20 (project configured for modern MSVC toolset)

The project is configured as a Visual Studio solution (`LargeMatrixMultiply.slnx`) and uses the MSVC toolchain. Precompiled headers are used (see `pch.h` / `pch.cpp`).

## Build

### Visual Studio

1. Open `LargeMatrixMultiply.slnx` in Visual Studio.
2. Select configuration (`Debug` or `Release`) and platform (`x64` recommended).
3. Build the solution (Build > Build Solution or Ctrl+Shift+B).

### Command line (MSBuild)

To build from a Developer Command Prompt or PowerShell with MSBuild available:

```powershell
# Build the main project (Release x64)
msbuild LargeMatrixMultiply.slnx /p:Configuration=Release /p:Platform=x64

# Build the tests project (Debug x64)
msbuild Tests\Tests.vcxproj /p:Configuration=Debug /p:Platform=x64
```

## Running tests

Recommended: use Visual Studio Test Explorer to run and debug tests in the `Tests` project.

Command-line options after building:

1. Locate the test binary (the project output folder under `Tests\x64\Debug` or the solution's output directory).
2. Run with `vstest.console.exe` (for .dll test assemblies) or execute the produced test binary directly:

```powershell
# Example (adjust path to actual test output):
vstest.console.exe .\Tests\x64\Debug\Tests.dll
# Or run native exe if produced:
.\Tests\x64\Debug\Tests.exe
```

If you prefer, simply open the solution in Visual Studio and run all tests via Test Explorer.

## Notes and best practices

- Large heap allocations: prefer `std::vector` (or other heap containers) for large temporary arrays rather than large stack allocations. This project uses heap allocation patterns to avoid stack overflows for big matrices (see project guidance in `.github/copilot-instructions.md`).
- The code uses precompiled headers for faster builds. If you add new translation units, include `pch.h` as the first include where appropriate.
- Target `x64` for large-matrix scenarios to avoid 32-bit address-space limits.

## Sparsity representation

This codebase provides sparse storage for both matrices and vectors. The on-disk/in-memory layouts used by the classes are:

- SparseMatrix (CSR / CSC):
	- Stored using three arrays: `m_values` (non-zero values), `m_indices` (row or column indices), and `m_pointers` (offsets into the previous two arrays).
	- When the matrix is row-major (CSR): `m_pointers` has length `M+1` and each row i's non-zero elements are in the range `[m_pointers[i], m_pointers[i+1])`. `m_indices` stores column indices for those values.
	- When the matrix is column-major (CSC): `m_pointers` has length `N+1` and each column j's non-zero elements are in the range `[m_pointers[j], m_pointers[j+1])`. `m_indices` stores row indices for those values.
	- The implementation keeps indices sorted within each row/column, uses binary search for element lookup, and updates `m_pointers` when inserting or removing non-zero entries.
	- Conversion helpers are provided (`ConvertCSRtoCSC()` / `ConvertCSCtoCSR()`) to flip between CSR and CSC representations.
	- Dense ↔ sparse conversion helpers are available (`ToDense()` and constructors that accept dense arrays).

- SparseVector:
	- Stored using two arrays: `m_indices` (sorted indices of non-zero elements) and `m_data` (corresponding non-zero values).
	- Element access uses binary search on `m_indices` and returns `T{}` when an index is not present.
	- Dot-products are implemented efficiently:
		- Sparse × sparse: two-pointer merge-style iteration over `m_indices`.
		- Sparse × dense: iterate non-zero entries of the sparse vector and sample the dense vector at those indices.

Notes:

- The project treats `T{}` (the default value for the element type) as the logical "zero" and stores only values != `T{}` in sparse containers.
- See the implementations for details and exact APIs in `Matrix.h` and `Vector.h`.

## Sparse operations: matrix-vector and matrix-matrix

Matrix-vector and matrix-matrix products are implemented to exploit the CSR/CSC and sparse-vector layouts efficiently:

- Matrix × Vector (sparse matrix):
	- CSR (row-major) uses `RightMultiply(const ScalarVector&)` semantics: for each row i iterate indices `j` in `[m_pointers[i], m_pointers[i+1])` and accumulate `m_values[j] * vec[m_indices[j]]` into `result[i]`.
	- CSC (column-major) uses `LeftMultiply(const ScalarVector&)` semantics: for each column j iterate indices `i` in `[m_pointers[j], m_pointers[j+1])` and accumulate `m_values[i] * vec[m_indices[i]]` into `result[j]` (or into destination rows when multiplying from the left).
	- Complexity: O(nnz) where nnz is the number of stored non-zero values.

- Matrix × Matrix:
	- The code expects one operand to be in CSR (row-major) and the other in CSC (column-major) for the most efficient multiply paths. Concretely, `SparseMatrix::RightMultiply(const ScalarMatrix& other)` requires `this` to be row-major and `other` to be column-major — this lets the implementation iterate a row of `A` and a column of `B` without scanning full rows/columns.
	- Sparse × Dense: for each row of the sparse matrix, iterate its non-zero entries and multiply-accumulate against the corresponding (dense) column entries of the other matrix (stored column-major for cache-friendly access). Complexity roughly O(nnz * K) for producing an M×K result.
	- Sparse × Sparse: when both matrices are sparse and stored in the appropriate complementary formats, the implementation performs a two-pointer merge between the sorted index lists of a row (from CSR) and a column (from CSC) to find matching indices and accumulate products. This avoids hashing or random lookups and runs in time proportional to the sum of degrees of the involved row/column pairs.
	- The code also implements `RightMultiply(const SparseMatrix&)` which uses the two-pointer technique to multiply matching non-zero index lists and store the summed result into the dense result buffer (a `ScalarMatrix`), as a simple, robust approach.

- Sparse Vector dot-products:
	- `dot(const SparseVector& a, const SparseVector& b)`: two‑pointer merge over the sorted `m_indices` arrays of both operands to find matching indices and accumulate products. Complexity O(nnz(a) + nnz(b)); minimal extra memory traffic.
	- `dot(const SparseVector& a, const ScalarVector& b)`: iterate `a.m_indices` / `a.m_data`, sample `b` at each index and accumulate. Complexity O(nnz(a)); avoids scanning the full dense vector.
	- Implementation notes: ensure `m_indices` are sorted to enable linear merges; avoid per-element binary searches in hot loops; prefer the two‑pointer merge when both operands are sparse.

Notes and trade-offs:

- Requiring one operand in CSR and the other in CSC minimizes random memory access: rows of A are contiguous slices of `m_values`/`m_indices` and columns of B are contiguous slices when stored CSC.
- Converting between CSR and CSC (via `ConvertCSRtoCSC()` / `ConvertCSCtoCSR()`) is supported but incurs O(nnz) work and temporary memory; avoid repeated conversions in hot code paths.
- For very large, highly-sparse matrices, consider algorithms that produce sparse outputs directly (assembly into CSR/CSC) instead of materializing dense intermediate results.

See `Matrix.h` and `Vector.h` for the exact method names and implementations used (`RightMultiply`, `LeftMultiply`, two-pointer merge, etc.).

## Benchmarks and Sparsity Analysis

The repository includes simple microbenchmarks for sparse and dense matrix/vector kernels. Below are representative results and a short interpretation of their implications for when sparse formats help. Benchmark is performed on Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz, 3696 Mhz, 10 Core(s), 20 Logical Processor(s).

Raw benchmark output (measured in microseconds):

```
mat*vec sparsity 0.010000: dense micro=2422.000000, sparse micro=85.000000, speedup=28.494118x
mat*mat sparsity 0.010000: dense dense micro=257576.000000, sparse*scalar micro=23012.000000, sparse*sparse micro=31521.000000
	.speedups (dense/sparse): scalar-mat=11.193117x, sparse-sparse=8.171568x
mat*vec sparsity 0.050000: dense micro=2390.000000, sparse micro=406.000000, speedup=5.886700x
mat*mat sparsity 0.050000: dense dense micro=256678.000000, sparse*scalar micro=84767.000000, sparse*sparse micro=170114.000000
	.speedups (dense/sparse): scalar-mat=3.028042x, sparse-sparse=1.508859x
mat*vec sparsity 0.100000: dense micro=2363.000000, sparse micro=787.000000, speedup=3.002541x
mat*mat sparsity 0.100000: dense dense micro=255914.000000, sparse*scalar micro=161375.000000, sparse*sparse micro=337035.000000
	.speedups (dense/sparse): scalar-mat=1.585834x, sparse-sparse=0.759310x
mat*vec sparsity 0.250000: dense micro=2361.000000, sparse micro=1980.000000, speedup=1.192424x
mat*mat sparsity 0.250000: dense dense micro=256298.000000, sparse*scalar micro=397345.000000, sparse*sparse micro=852453.000000
	.speedups (dense/sparse): scalar-mat=0.645026x, sparse-sparse=0.300659x
```

Summary and interpretation:

- **SpMV (matrix × vector):** At very low density (1% nonzeros) the sparse implementation is dramatically faster (×28 in this run). This is typical for SpMV when the dense kernel is memory‑bound: skipping zeros drastically reduces memory traffic. As density increases the advantage shrinks because indirect indexing, diminished cache reuse, and extra per‑nonzero overhead reduce performance; the provided data show sparse wins up to roughly 25% density and becomes marginal near 30%.

- **SpMM (matrix × matrix):** Sparse matrix–matrix multiplication only outperforms the dense path at very low densities in these measurements (1% and in some cases 5%). For moderate densities (≥10%) the sparse implementations are slower: indexing indirection, irregular memory access, and lower arithmetic intensity make dense BLAS-style code faster. This matches common SpMM findings in literature.

- **Why sparse can be slower at higher density:** Compressed formats add indirection and reduce contiguous memory access. When many entries are nonzero the per‑element overhead (index checks, pointer chasing, accumulation into scattered locations) outweighs benefits from skipping zeros.

- **Practical recommendations:**
  - Use sparse kernels for very low-density matrices (single-digit percent nonzeros).
  - For moderate densities prefer dense BLAS (or fallback to dense after a density threshold).
  - Consider block‑sparse, tiled, or hybrid formats (or convert to block dense) to recover cache locality when sparsity has structured nonzeros.
  - Measure on your target hardware and problem sizes; hardware memory bandwidth, cache sizes, and BLAS implementation quality strongly affect crossover points.

These notes are intended as guidance; the raw numbers above are from a microbenchmark run and should be interpreted relative to your machine and build configuration.

### Sparse vector dot-product microbenchmarks

Raw benchmark output (vector dot products, measured in milliseconds):

```
dot(SparseVector, SparseVector) vs dot(ScalarVector, ScalarVector)
	Vector Size: 10000, Sparsity: 0.010000
	Sparse ms: 1 ms, 	Dense ms: 113 ms
	Speedup: 113.000000x
dot(SparseVector, SparseVector) vs dot(ScalarVector, ScalarVector)
	Vector Size: 10000, Sparsity: 0.050000
	Sparse ms: 9 ms, 	Dense ms: 114 ms
	Speedup: 12.666667x
dot(SparseVector, SparseVector) vs dot(ScalarVector, ScalarVector)
	Vector Size: 10000, Sparsity: 0.100000
	Sparse ms: 19 ms, 	Dense ms: 114 ms
	Speedup: 6.000000x
dot(SparseVector, SparseVector) vs dot(ScalarVector, ScalarVector)
	Vector Size: 10000, Sparsity: 0.250000
	Sparse ms: 47 ms, 	Dense ms: 113 ms
	Speedup: 2.404255x

dot(SparseVector, ScalarVector) vs dot(ScalarVector, ScalarVector)
	Vector Size: 10000, Sparsity: 0.010000
	Sparse ms: 1 ms, 	Dense ms: 117 ms
	Speedup: 117.000000x
dot(SparseVector, ScalarVector) vs dot(ScalarVector, ScalarVector)
	Vector Size: 10000, Sparsity: 0.050000
	Sparse ms: 6 ms, 	Dense ms: 116 ms
	Speedup: 19.333333x
dot(SparseVector, ScalarVector) vs dot(ScalarVector, ScalarVector)
	Vector Size: 10000, Sparsity: 0.100000
	Sparse ms: 13 ms, 	Dense ms: 115 ms
	Speedup: 8.846154x
dot(SparseVector, ScalarVector) vs dot(ScalarVector, ScalarVector)
	Vector Size: 10000, Sparsity: 0.250000
	Sparse ms: 31 ms, 	Dense ms: 116 ms
	Speedup: 3.741935x
```

Observations:

- These dot-product microbenchmarks show very large speedups for sparse vectors at low densities (×100+ at 1% sparsity). Even at 25% nonzeros the sparse dot remains faster (2–4×) for the measured vector size (10k).
- Dot-product operations are especially friendly to sparse formats because the sparse algorithm only iterates the nonzero entries and performs O(nnz) work rather than scanning the full dense vector — this greatly reduces memory traffic and leads to large wins when nnz is small relative to the vector length.

Implementation notes (how this repo implements dot products):

- `dot(const SparseVector& a, const SparseVector& b)`: two‑pointer merge over the sorted `m_indices` arrays of both operands. Complexity O(nnz(a) + nnz(b)) and minimal overhead per matching index.
- `dot(const SparseVector& a, const ScalarVector& b)` (sparse × dense): iterate `a.m_indices` and `a.m_data`, sample `b` at each index, and accumulate. Complexity O(nnz(a)); memory accesses are reads from the dense vector at the sparse indices.
- Avoid per-element binary search when possible; ensure `m_indices` are sorted and use the linear two-pointer merge for sparse×sparse to keep overhead low.

Practical guidance:

- For pure dot-products the sparsity crossover is usually much higher than for SpMV/SpMM — sparse vectors remain beneficial at substantially larger densities because the operation is a single-pass reduction with no scattered writes.
- Use sparse dot for feature vectors, embeddings with many zeros, or other high-dimensional sparse signals; for small vectors or very high densities ( > ~50%) measure and consider using dense routines.
- If you frequently need mixed operations (dense & sparse), ensure efficient SIMD/BLAS fallbacks and consider converting between representations at runtime based on density heuristics.

## Contributing

- Follow the existing code style (modern C++, avoid macros where possible).
- Add unit tests to `Tests/` for any new functionality.
- Run both `Debug` and `Release` builds when validating changes.

## License

This repository includes a `LICENSE.txt` file — see it for license details.