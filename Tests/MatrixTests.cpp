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
        }
    };

    TEST_CLASS(SparseMatrixTests)
    {
    public:
        TEST_METHOD(TestSparseMatrixCreation_Default)
        {
            auto matrix = std::make_unique<SparseMatrix<float, 3, 4>>();
            Assert::IsNotNull(matrix.get(), L"Sparse matrix should be created successfully.");
        }

        
    };
}