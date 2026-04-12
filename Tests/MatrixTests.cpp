#include "pch.h"
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MatrixTests
{
    TEST_CLASS(ScalarMatrixTests)
    {
    public:
        TEST_METHOD(TestScalarMatrixCreation_Default)
        {
            // Create a 3x4 matrix of floats
            auto matrix = std::make_unique<ScalarMatrix<float, 3, 4>>();
            Assert::IsNotNull(matrix.get(), L"Matrix should be created successfully.");
            Assert::AreEqual(false, matrix->IsColumnMajor(), L"Default storage format should be row-major.");
        }
    };

    TEST_CLASS(SparseMatrixTests)
    {
    public:
        TEST_METHOD(TestSparseMatrixCreation_Default)
        {
            auto matrix = std::make_unique<SparseMatrix<float, 3, 4>>();
            Assert::IsNotNull(matrix.get(), L"Sparse matrix should be created successfully.");
            Assert::AreEqual(false, matrix->IsColumnMajor(), L"Default storage format should be row-major.");
            
            Assert::AreEqual(uint32_t(3), (uint32_t)matrix->GetNumRows(), L"Number of matrix rows should be 3.");
            Assert::AreEqual(uint32_t(4), (uint32_t)matrix->GetNumCols(), L"Number of matrix cols should be 4.");
            Assert::AreEqual(uint32_t(0), (uint32_t)matrix->GetNonZeroCount(), L"Newly created sparse matrix should have zero non-zero entries.");
        }

        
    };
}