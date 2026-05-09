# LargeMatrixMultiply

A small, focused C++ project implementing matrix and vector primitives and unit tests used by the DeepLearningGraphics suite.

## Project summary

This repository contains native C++ implementations for matrix and vector operations used for large-scale numerical work. The code is organized as a Visual Studio project and includes a small unit-test project under `Tests/`.

Key source files:

- `Matrix.cpp`, `Matrix.h` — matrix operations and helpers
- `Vector.cpp`, `Vector.h` — vector utilities
- `Utilities.cpp`, `Utilities.h` — supporting utilities and helpers
- `Tests/` — unit tests (GoogleTest or custom test harness via Visual Studio Test project)

## Requirements

- Windows 10 or later
- Visual Studio 2019/2022 with C++ workload (MSVC)
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
	- CSR (row-major) uses `RightMultiply(const ScalarVector&)` / `RightMultiply(const SparseVector&)` semantics: for each row i iterate indices `j` in `[m_pointers[i], m_pointers[i+1])` and accumulate `m_values[j] * vec[m_indices[j]]` into `result[i]`.
	- CSC (column-major) uses `LeftMultiply(const ScalarVector&)` semantics: for each column j iterate indices `i` in `[m_pointers[j], m_pointers[j+1])` and accumulate `m_values[i] * vec[m_indices[i]]` into `result[j]` (or into destination rows when multiplying from the left).
	- Complexity: O(nnz) where nnz is the number of stored non-zero values.

- Matrix × Vector (sparse vector):
	- The `SparseVector` stores `(m_indices, m_data)` with `m_indices` sorted. Element access is binary-search based, and dot products use a two-pointer merge when both operands are sparse, otherwise the sparse operand iterates its non-zero entries and samples the dense vector.

- Matrix × Matrix:
	- The code expects one operand to be in CSR (row-major) and the other in CSC (column-major) for the most efficient multiply paths. Concretely, `SparseMatrix::RightMultiply(const ScalarMatrix& other)` requires `this` to be row-major and `other` to be column-major — this lets the implementation iterate a row of `A` and a column of `B` without scanning full rows/columns.
	- Sparse × Dense: for each row of the sparse matrix, iterate its non-zero entries and multiply-accumulate against the corresponding (dense) column entries of the other matrix (stored column-major for cache-friendly access). Complexity roughly O(nnz * K) for producing an M×K result.
	- Sparse × Sparse: when both matrices are sparse and stored in the appropriate complementary formats, the implementation performs a two-pointer merge between the sorted index lists of a row (from CSR) and a column (from CSC) to find matching indices and accumulate products. This avoids hashing or random lookups and runs in time proportional to the sum of degrees of the involved row/column pairs.
	- The code also implements `RightMultiply(const SparseMatrix&)` which uses the two-pointer technique to multiply matching non-zero index lists and store the summed result into the dense result buffer (a `ScalarMatrix`), as a simple, robust approach.

Notes and trade-offs:

- Requiring one operand in CSR and the other in CSC minimizes random memory access: rows of A are contiguous slices of `m_values`/`m_indices` and columns of B are contiguous slices when stored CSC.
- Converting between CSR and CSC (via `ConvertCSRtoCSC()` / `ConvertCSCtoCSR()`) is supported but incurs O(nnz) work and temporary memory; avoid repeated conversions in hot code paths.
- For very large, highly-sparse matrices, consider algorithms that produce sparse outputs directly (assembly into CSR/CSC) instead of materializing dense intermediate results.

See `Matrix.h` and `Vector.h` for the exact method names and implementations used (`RightMultiply`, `LeftMultiply`, two-pointer merge, etc.).

## Contributing

- Follow the existing code style (modern C++, avoid macros where possible).
- Add unit tests to `Tests/` for any new functionality.
- Run both `Debug` and `Release` builds when validating changes.

## License

This repository includes a `LICENSE.txt` file — see it for license details.

## Contact

Part of the DeepLearningGraphics project suite. For questions, open an issue or contact the maintainers listed in the repository metadata.

---

This README was updated to add build and test instructions, file references, and practical notes for working on the project.
