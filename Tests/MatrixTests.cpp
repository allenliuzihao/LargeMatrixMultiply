#include "pch.h"
#include "CppUnitTest.h"
#include <random>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MatrixTests
{
    TEST_CLASS(SparseMatrixTests)
    {
    private:
        template <size_t M, size_t N>
        void CheckMatrixEqual(const ScalarMatrix<float, M, N>& before, const ScalarMatrix<float, M, N>& after)
        {
            Assert::AreEqual(before.IsColumnMajor(), after.IsColumnMajor(), L"Storage format mismatch");
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    float b = before.GetElement(i, j);
                    float a = after.GetElement(i, j);
                    Assert::AreEqual((double)b, (double)a, 1e-6, L"CSR<->CSC data mismatch");
                }
            }
        }

    public:
        TEST_METHOD(TestScalarMatrixCreation_Default)
        {
            // Create a 3x4 matrix of floats
            auto matrix = std::make_unique<ScalarMatrix<float, 3, 4>>();
            Assert::IsNotNull(matrix.get(), L"Matrix should be created successfully.");
            Assert::AreEqual(false, matrix->IsColumnMajor(), L"Default storage format should be row-major.");
        }

        TEST_METHOD(TestSparseCSRCSCConversionPreservesData)
        {
            // small deterministic matrix to validate CSR <-> CSC conversions
            constexpr size_t M = 7, N = 9;
            std::vector<float> data(M * N, 0.0f);
            // set a reproducible pattern
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    if (((i * 31 + j * 17) % 5) == 0) data[i * N + j] = static_cast<float>((i + j) % 10 + 1);

            // build CSR sparse matrix from row-major data
            auto csr = std::make_unique<SparseMatrix<float, M, N>>(data.data(), data.size(), /*isColumnMajor=*/false);
            auto before = csr->ToDense();

            // convert to CSC and back
            csr->FlipStorageFormat(); // now CSC
            csr->FlipStorageFormat(); // back to CSR
            auto after = csr->ToDense();
            Assert::AreEqual(before->TotalMemoryBytes(), after->TotalMemoryBytes(), L"CSR->CSC->CSR size mismatch");

            // check if matrices before and after are approximately equal (allowing for floating point precision issues)
            CheckMatrixEqual(*before, *after);
                
            // also verify constructing directly as CSC (from column-major buffer) yields same row-major export
            std::vector<float> colData(M * N);
            for (size_t r = 0; r < M; ++r)
                for (size_t c = 0; c < N; ++c)
                    colData[c * M + r] = data[r * N + c];

            auto csc = std::make_unique<SparseMatrix<float, M, N>>(colData.data(), colData.size(), /*isColumnMajor=*/true);
            csc->FlipStorageFormat();
            after = csc->ToDense();
            CheckMatrixEqual(*before, *after);
        }

        TEST_METHOD(TestSparseDenseConversionPreservesData)
        {
            // validate constructing sparse from dense ScalarMatrix and back preserves data
            constexpr size_t M = 6, N = 8;
            ScalarMatrix<float, M, N> dense(false);
            // populate dense with a pattern
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                {
                    float v = (((int)i * 7 + (int)j * 13) % 6 == 0) ? static_cast<float>((i + j) % 9 + 1) : 0.0f;
                    dense.SetElement(i, j, v);
                }

            // build sparse CSR from dense
            auto sparseFromDense = std::make_unique<SparseMatrix<float, M, N>>(dense, /*isColumnMajor=*/false);
            auto sparseAsRow = sparseFromDense->ToDense();
            CheckMatrixEqual(dense, *sparseAsRow);

            // build sparse CSC from dense (column-major) and verify
            auto sparseFromDenseCSC = std::make_unique<SparseMatrix<float, M, N>>(dense, /*isColumnMajor=*/true);
            auto sparseCSCAsRow = sparseFromDenseCSC->ToDense();
            dense.FlipStorageFormat(); // flip dense to column-major for comparison
            CheckMatrixEqual(dense, *sparseCSCAsRow);
        }

        TEST_METHOD(TestMatrixMatrixRightMultiply_Correctness)
        {
            // Small sizes to validate correctness of matrix-matrix RightMultiply overloads
            constexpr size_t M = 10, N = 12, K = 8;
            std::vector<float> Adata(M * N, 0.0f);
            std::vector<float> Bdata(N * K, 0.0f);

            // deterministic values
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    Adata[i * N + j] = static_cast<float>((i + 1) * (j + 2) % 7 + 1);

            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < K; ++j)
                    Bdata[i * K + j] = static_cast<float>((i + 3) * (j + 5) % 11 + 1);

            // reference dense multiplication (row-major arrays)
            std::vector<float> ref(M * K, 0.0f);
            for (size_t i = 0; i < M; ++i)
                for (size_t k = 0; k < K; ++k)
                {
                    float s = 0.0f;
                    for (size_t n = 0; n < N; ++n) s += Adata[i * N + n] * Bdata[n * K + k];
                    ref[i * K + k] = s;
                }

            // Scalar * Scalar (A row-major, B column-major)
            ScalarMatrix<float, M, N> Arow(false);
            ScalarMatrix<float, N, K> Bcol(true);
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    Arow.SetElement(i, j, Adata[i * N + j]);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < K; ++j)
                    Bcol.SetElement(i, j, Bdata[i * K + j]);

            auto denseRes = Arow.RightMultiply(Bcol);
            for (size_t i = 0; i < M; ++i)
                for (size_t k = 0; k < K; ++k)
                    Assert::AreEqual((double)ref[i * K + k], (double)(*denseRes).GetElement(i, k), 1e-4, L"ScalarMatrix x ScalarMatrix result mismatch");

            // Sparse (CSR) A * Scalar (B column-major)
            auto sparseA = std::make_unique<SparseMatrix<float, M, N>>(Adata.data(), Adata.size(), /*isColumnMajor=*/false);
            auto sparseDenseRes = sparseA->RightMultiply(Bcol);
            for (size_t i = 0; i < M; ++i)
                for (size_t k = 0; k < K; ++k)
                    Assert::AreEqual((double)ref[i * K + k], (double)sparseDenseRes->GetElement(i, k), 1e-4, L"SparseMatrix x ScalarMatrix result mismatch");

            // Sparse * Sparse (A CSR, B CSC)
            // build column-major buffer for B
            std::vector<float> BcolData(N * K);
            for (size_t r = 0; r < N; ++r)
                for (size_t c = 0; c < K; ++c)
                    BcolData[c * N + r] = Bdata[r * K + c];

            auto sparseB = std::make_unique<SparseMatrix<float, N, K>>(BcolData.data(), BcolData.size(), /*isColumnMajor=*/true);
            auto sparseSparseRes = sparseA->RightMultiply(*sparseB);
            for (size_t i = 0; i < M; ++i)
                for (size_t k = 0; k < K; ++k)
                    Assert::AreEqual((double)ref[i * K + k], (double)sparseSparseRes->GetElement(i, k), 1e-4, L"SparseMatrix x SparseMatrix result mismatch");
        }

        TEST_METHOD(TestMatrixMatrixRightMultiply_Performance)
        {
            // compare performance of ScalarMatrix x ScalarMatrix vs SparseMatrix x SparseMatrix
            constexpr size_t M = 512, N = 512, K = 512;
            const double sparsity = 0.05; // 5% non-zero

            std::mt19937 rng(2026);
            std::bernoulli_distribution keep(sparsity);
            std::uniform_real_distribution<float> valDist(1.0f, 5.0f);

            std::vector<float> Adata(M * N, 0.0f);
            std::vector<float> Bdata(N * K, 0.0f);
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    if (keep(rng)) Adata[i * N + j] = valDist(rng);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < K; ++j)
                    if (keep(rng)) Bdata[i * K + j] = valDist(rng);

            // build scalar matrices
            ScalarMatrix<float, M, N> Arow(false);
            ScalarMatrix<float, N, K> Bcol(true);
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    Arow.SetElement(i, j, Adata[i * N + j]);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < K; ++j)
                    Bcol.SetElement(i, j, Bdata[i * K + j]);

            // time dense scalar x scalar
            auto t0 = std::chrono::high_resolution_clock::now();
            auto denseRes = Arow.RightMultiply(Bcol);
            auto t1 = std::chrono::high_resolution_clock::now();

            // build A as row major sparse matrices
            auto sparseA = std::make_unique<SparseMatrix<float, M, N>>(Arow, /*isColumnMajor=*/false);
            // build B as column-major sparse matrices
            auto sparseB = std::make_unique<SparseMatrix<float, N, K>>(Bcol, /*isColumnMajor=*/true);

            auto t2 = std::chrono::high_resolution_clock::now();
            auto sparseRes = sparseA->RightMultiply(*sparseB);
            auto t3 = std::chrono::high_resolution_clock::now();

            auto denseMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            auto sparseMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

            // verify equality of results
            for (size_t i = 0; i < M; ++i)
                for (size_t k = 0; k < K; ++k)
                    Assert::AreEqual((double)denseRes->GetElement(i, k), (double)sparseRes->GetElement(i, k), 1e-3, L"Matrix-matrix multiply result mismatch");

            Logger::WriteMessage(("Dense matrix-matrix ms: " + std::to_string(denseMs) + "\n").c_str());
            Logger::WriteMessage(("Sparse matrix-matrix ms: " + std::to_string(sparseMs) + "\n").c_str());

            Assert::IsTrue(sparseMs < denseMs, L"Sparse matrix-matrix multiply should be faster than dense for 30% sparsity in this test.");
        }

        TEST_METHOD(TestLargeSparseLeftRightFormatConversion)
        {
            // large matrix with 30% non-zero entries
            constexpr size_t M = 2000, N = 2000; // large but reasonable for CI on modern machines
            std::vector<float> data(M * N, 0.0f);

            std::mt19937 rng(12345);
            std::bernoulli_distribution keep(0.30); // 30% non-zero
            std::uniform_real_distribution<float> valDist(1.0f, 5.0f);

            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    if (keep(rng)) data[i * N + j] = valDist(rng);
                }
            }

            // right-multiply: A (MxN) * x (N) -> y (M)
            ScalarVector<float, N> vecN;
            for (size_t j = 0; j < N; ++j) vecN[j] = static_cast<float>((j % 7) + 1);

            // dense right multiply
            std::vector<float> denseRight(M, 0.0f);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < M; ++i)
            {
                float s = 0.0f;
                for (size_t j = 0; j < N; ++j) s += data[i * N + j] * vecN[j];
                denseRight[i] = s;
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            // build sparse CSR and time right multiply
            auto sparse = std::make_unique<SparseMatrix<float, M, N>>(data.data(), data.size(), /*isColumnMajor=*/false);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto sparseRightPtr = sparse->RightMultiply(vecN);
            auto t3 = std::chrono::high_resolution_clock::now();

            auto denseRightDur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
            auto sparseRightDur = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

            // validate results
            for (size_t i = 0; i < M; ++i) Assert::AreEqual((double)denseRight[i], (double)(*sparseRightPtr)[i], 1e-3, L"Right multiply mismatch");

            Logger::WriteMessage(("Dense right multiply ms: " + std::to_string(denseRightDur.count()) + "\n").c_str());
            Logger::WriteMessage(("Sparse right multiply ms: " + std::to_string(sparseRightDur.count()) + "\n").c_str());
            Assert::IsTrue(sparseRightDur.count() < denseRightDur.count(), L"Sparse right multiply should be faster than dense for 30% non-zero.");

            // left-multiply: x (M) * A (MxN) -> y (N)
            ScalarVector<float, M> vecM;
            for (size_t i = 0; i < M; ++i) vecM[i] = static_cast<float>((i % 11) + 1);

            // dense left multiply
            std::vector<float> denseLeft(N, 0.0f);
            auto t4 = std::chrono::high_resolution_clock::now();
            for (size_t j = 0; j < N; ++j)
            {
                float s = 0.0f;
                for (size_t i = 0; i < M; ++i) s += data[i * N + j] * vecM[i];
                denseLeft[j] = s;
            }
            auto t5 = std::chrono::high_resolution_clock::now();

            // build sparse CSC (column-major) for left multiply
            std::vector<float> colData(M * N);
            for (size_t r = 0; r < M; ++r)
                for (size_t c = 0; c < N; ++c)
                    colData[c * M + r] = data[r * N + c];

            sparse->FlipStorageFormat();
            auto t6 = std::chrono::high_resolution_clock::now();
            auto sparseLeftPtr = sparse->LeftMultiply(vecM);
            auto t7 = std::chrono::high_resolution_clock::now();

            auto denseLeftDur = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4);
            auto sparseLeftDur = std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6);

            for (size_t j = 0; j < N; ++j) Assert::AreEqual((double)denseLeft[j], (double)(*sparseLeftPtr)[j], 1e-3, L"Left multiply mismatch");

            Logger::WriteMessage(("Dense left multiply ms: " + std::to_string(denseLeftDur.count()) + "\n").c_str());
            Logger::WriteMessage(("Sparse left multiply ms: " + std::to_string(sparseLeftDur.count()) + "\n").c_str());
            Assert::IsTrue(sparseLeftDur.count() < denseLeftDur.count(), L"Sparse left multiply should be faster than dense for 30% non-zero.");
        }

        TEST_METHOD(TestSparseMatrixCreation_Default)
        {
            auto matrix = std::make_unique<SparseMatrix<float, 3, 4>>();
            Assert::IsNotNull(matrix.get(), L"Sparse matrix should be created successfully.");
            Assert::AreEqual(false, matrix->IsColumnMajor(), L"Default storage format should be row-major.");

            Assert::AreEqual(uint32_t(3), (uint32_t)matrix->GetNumRows(), L"Number of matrix rows should be 3.");
            Assert::AreEqual(uint32_t(4), (uint32_t)matrix->GetNumCols(), L"Number of matrix cols should be 4.");
            Assert::AreEqual(uint32_t(0), (uint32_t)matrix->GetNonZeroCount(), L"Newly created sparse matrix should have zero non-zero entries.");
        }
        TEST_METHOD(TestSparseRightMultiplyMatchesDense)
        {
            // A: 4x5 matrix (row-major data array)
            constexpr size_t M = 4, N = 5;
            std::vector<float> data(M * N, 0.0f);
            // set some non-zero elements
            data[0 * N + 1] = 2.0f;
            data[1 * N + 0] = -1.0f;
            data[2 * N + 4] = 3.5f;
            data[3 * N + 2] = 4.25f;

            // build a dense input vector of size N
            ScalarVector<float, N> vec;
            for (size_t i = 0; i < N; ++i) vec[i] = static_cast<float>(i + 1);

            // expected dense result: result[i] = sum_j A[i,j]*vec[j]
            std::array<float, M> expected{};
            for (size_t i = 0; i < M; ++i)
            {
                float s = 0.0f;
                for (size_t j = 0; j < N; ++j) s += data[i * N + j] * vec[j];
                expected[i] = s;
            }

            // create sparse matrix from row-major data
            auto sparse = std::make_unique<SparseMatrix<float, M, N>>(data.data(), data.size(), /*isColumnMajor=*/false);
            Assert::AreEqual(uint32_t(4), (uint32_t)sparse->GetNumRows(), L"rows");
            Assert::AreEqual(uint32_t(5), (uint32_t)sparse->GetNumCols(), L"cols");

            auto res = sparse->RightMultiply(vec);
            for (size_t i = 0; i < M; ++i)
            {
                Assert::AreEqual((double)expected[i], (double)(*res)[i], 1e-5, L"RightMultiply result mismatch");
            }
        }

        TEST_METHOD(TestSparseLeftMultiplyMatchesDense)
        {
            // A: 4x5 matrix, we will construct column-major sparse matrix to test LeftMultiply
            constexpr size_t M = 4, N = 5;
            std::vector<float> data(M * N, 0.0f);
            data[0 * N + 1] = 2.0f;
            data[1 * N + 0] = -1.0f;
            data[2 * N + 4] = 3.5f;
            data[3 * N + 2] = 4.25f;

            // create column-major layout from same logical matrix
            std::vector<float> colData(M * N, 0.0f);
            for (size_t r = 0; r < M; ++r)
                for (size_t c = 0; c < N; ++c)
                    colData[c * M + r] = data[r * N + c];

            // input vector of size M for left-multiply
            ScalarVector<float, M> vec;
            for (size_t i = 0; i < M; ++i) vec[i] = static_cast<float>(i + 1);

            // expected dense result: for each column j: sum_i A[i,j]*vec[i]
            std::array<float, N> expected{};
            for (size_t j = 0; j < N; ++j)
            {
                float s = 0.0f;
                for (size_t i = 0; i < M; ++i) s += data[i * N + j] * vec[i];
                expected[j] = s;
            }

            // build sparse column-major matrix
            auto sparseCol = std::make_unique<SparseMatrix<float, M, N>>(colData.data(), colData.size(), /*isColumnMajor=*/true);
            auto res = sparseCol->LeftMultiply(vec);
            for (size_t j = 0; j < N; ++j)
            {
                Assert::AreEqual((double)expected[j], (double)(*res)[j], 1e-5, L"LeftMultiply result mismatch");
            }
        }

        TEST_METHOD(TestFlipStorageBehaviorAndExceptions)
        {
            // ScalarMatrix flip toggles storage and affects supported operations
            auto scalar = std::make_unique<ScalarMatrix<float, 3, 3>>();
            Assert::IsFalse(scalar->IsColumnMajor());
            scalar->FlipStorageFormat();
            Assert::IsTrue(scalar->IsColumnMajor());
            // RightMultiply should now throw for column-major
            ScalarVector<float, 3> v;
            Assert::ExpectException<std::runtime_error>([&](){ scalar->RightMultiply(v); });

            // SparseMatrix flip toggles storage and affects supported operations
            std::vector<float> data(9, 0.0f);
            data[0] = 1.0f; data[4] = 2.0f;
            auto sparse = std::make_unique<SparseMatrix<float, 3, 3>>(data.data(), data.size(), /*isColumnMajor=*/false);
            Assert::IsFalse(sparse->IsColumnMajor());
            sparse->FlipStorageFormat();
            Assert::IsTrue(sparse->IsColumnMajor());
            Assert::ExpectException<std::runtime_error>([&](){ scalar->RightMultiply(v); });
        }

        TEST_METHOD(TestSparseMemoryIsLessThanDenseWhenSparse)
        {
            constexpr size_t M = 16, N = 16;
            std::vector<float> data(M * N, 0.0f);
            // make matrix very sparse: few entries
            data[0 * N + 0] = 1.0f;
            data[5 * N + 3] = 2.0f;
            data[10 * N + 15] = 3.0f;

            auto dense = std::make_unique<ScalarMatrix<float, M, N>>();
            auto sparse = std::make_unique<SparseMatrix<float, M, N>>(data.data(), data.size(), /*isColumnMajor=*/false);

            size_t denseBytes = dense->TotalMemoryBytes();
            size_t sparseBytes = sparse->TotalMemoryBytes();

            // print memory usage for informational purposes
            Logger::WriteMessage(("Dense matrix memory usage: " + std::to_string(denseBytes) + " bytes\n").c_str());
            Logger::WriteMessage(("Sparse matrix memory usage: " + std::to_string(sparseBytes) + " bytes\n").c_str());

            Assert::IsTrue(sparseBytes < denseBytes, L"Sparse representation should use less memory than dense when matrix is sparse.");
        }

        TEST_METHOD(TestSparsePerformanceSpeedupAgainstDense)
        {
            // Compare time to multiply a matrix by a vector using dense iteration vs sparse representation
            constexpr size_t M = 400, N = 400; // moderate size
            std::vector<float> data(M * N, 0.0f);
            // make ~1% non-zero
            for (size_t i = 0; i < M * N; ++i)
            {
                if ((i % 100) == 0) data[i] = static_cast<float>((i % 13) + 1);
            }

            // input vector
            ScalarVector<float, N> vec;
            for (size_t i = 0; i < N; ++i) vec[i] = static_cast<float>((i % 7) + 1);

            // time dense multiplication (naive row-major)
            std::vector<float> denseRes(M, 0.0f);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < M; ++i)
            {
                float s = 0.0f;
                for (size_t j = 0; j < N; ++j) s += data[i * N + j] * vec[j];
                denseRes[i] = s;
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            // build sparse CSR and time sparse multiplication
            auto sparse = std::make_unique<SparseMatrix<float, M, N>>(data.data(), data.size(), /*isColumnMajor=*/false);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto sparseResPtr = sparse->RightMultiply(vec);
            auto t3 = std::chrono::high_resolution_clock::now();

            auto denseDur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
            auto sparseDur = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);

            // validate results match
            for (size_t i = 0; i < M; ++i)
            {
                Assert::AreEqual((double)denseRes[i], (double)(*sparseResPtr)[i], 1e-3, L"Performance test result mismatch");
            }

            // print timings for informational purposes
            Logger::WriteMessage(("Dense multiplication time: " + std::to_string(denseDur.count()) + " microseconds\n").c_str());
            Logger::WriteMessage(("Sparse multiplication time: " + std::to_string(sparseDur.count()) + " microseconds\n").c_str());
            // sparse should be faster (or at least not slower) for this sparsity
            Assert::IsTrue(sparseDur.count() < denseDur.count() * 1.1, L"Sparse multiplication should be faster than dense iteration for very sparse matrices.");
        }
    };
}