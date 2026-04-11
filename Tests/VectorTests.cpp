#include "pch.h"
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace VectorTests
{
	TEST_CLASS(ScalarVectorTests)
	{
	public:
		TEST_METHOD(TestVectorDot_ZeroVectors)
		{
			// Dot product of two zero vectors should be 0
			std::unique_ptr<Vector<float, 3>> vec1 = std::make_unique<ScalarVector<float, 3>>();
			std::unique_ptr<Vector<float, 3>> vec2 = std::make_unique<ScalarVector<float, 3>>();
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(0.0f, dotProduct, L"Dot product of two zero vectors should be 0.");
		}

		TEST_METHOD(TestVectorDot_IdenticalVectors)
		{
			// Dot product of vector with itself: [1, 2, 3] · [1, 2, 3] = 1 + 4 + 9 = 14
			std::array<float, 3> data = { 1.0f, 2.0f, 3.0f };
			std::unique_ptr<Vector<float, 3>> vec1 = std::make_unique<ScalarVector<float, 3>>(data);
			float dotProduct = vec1->Dot(*vec1);
			Assert::AreEqual(14.0f, dotProduct, L"Dot product of [1,2,3] with itself should be 14.");
		}

		TEST_METHOD(TestVectorDot_OrthogonalVectors)
		{
			// Dot product of perpendicular vectors should be 0
			// [1, 0, 0] · [0, 1, 0] = 0
			std::array<float, 3> data1 = { 1.0f, 0.0f, 0.0f };
			std::array<float, 3> data2 = { 0.0f, 1.0f, 0.0f };
			std::unique_ptr<Vector<float, 3>> vec1 = std::make_unique<ScalarVector<float, 3>>(data1);
			std::unique_ptr<Vector<float, 3>> vec2 = std::make_unique<ScalarVector<float, 3>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(0.0f, dotProduct, L"Dot product of orthogonal vectors should be 0.");
		}

		TEST_METHOD(TestVectorDot_ParallelVectors)
		{
			// Dot product of parallel vectors: [1, 2, 3] · [2, 4, 6] = 2 + 8 + 18 = 28
			std::array<float, 3> data1 = { 1.0f, 2.0f, 3.0f };
			std::array<float, 3> data2 = { 2.0f, 4.0f, 6.0f };
			std::unique_ptr<Vector<float, 3>> vec1 = std::make_unique<ScalarVector<float, 3>>(data1);
			std::unique_ptr<Vector<float, 3>> vec2 = std::make_unique<ScalarVector<float, 3>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(28.0f, dotProduct, L"Dot product of [1,2,3] and [2,4,6] should be 28.");
		}

		TEST_METHOD(TestVectorDot_NegativeValues)
		{
			// Dot product with negative values: [1, -2, 3] · [2, 1, -1] = 2 - 2 - 3 = -3
			std::array<float, 3> data1 = { 1.0f, -2.0f, 3.0f };
			std::array<float, 3> data2 = { 2.0f, 1.0f, -1.0f };
			std::unique_ptr<Vector<float, 3>> vec1 = std::make_unique<ScalarVector<float, 3>>(data1);
			std::unique_ptr<Vector<float, 3>> vec2 = std::make_unique<ScalarVector<float, 3>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(-3.0f, dotProduct, L"Dot product of [1,-2,3] and [2,1,-1] should be -3.");
		}

		TEST_METHOD(TestVectorDot_IntegerType)
		{
			// Test with integer type: [1, 2, 3] · [1, 2, 3] = 14
			std::array<int, 3> data = { 1, 2, 3 };
			std::unique_ptr<Vector<int, 3>> vec1 = std::make_unique<ScalarVector<int, 3>>(data);
			float dotProduct = vec1->Dot(*vec1);
			Assert::AreEqual(14.0f, dotProduct, L"Dot product with int type [1,2,3] with itself should be 14.");
		}

		TEST_METHOD(TestVectorDot_UnitVectors)
		{
			// Dot product of unit vectors at different angles
			// [1, 0, 0] · [0, 0, 1] = 0 (perpendicular)
			std::array<float, 3> data1 = { 1.0f, 0.0f, 0.0f };
			std::array<float, 3> data2 = { 0.0f, 0.0f, 1.0f };
			std::unique_ptr<Vector<float, 3>> vec1 = std::make_unique<ScalarVector<float, 3>>(data1);
			std::unique_ptr<Vector<float, 3>> vec2 = std::make_unique<ScalarVector<float, 3>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(0.0f, dotProduct, L"Dot product of unit vectors [1,0,0] and [0,0,1] should be 0.");
		}

		TEST_METHOD(TestVectorDot_SmallValues)
		{
			// Dot product with small floating-point values
			// [0.1, 0.2, 0.3] · [0.1, 0.2, 0.3] = 0.01 + 0.04 + 0.09 = 0.14
			std::array<float, 3> data = { 0.1f, 0.2f, 0.3f };
			std::unique_ptr<Vector<float, 3>> vec1 = std::make_unique<ScalarVector<float, 3>>(data);
			float dotProduct = vec1->Dot(*vec1);
			Assert::AreEqual(0.14f, dotProduct, 0.0001f, L"Dot product of [0.1,0.2,0.3] with itself should be ~0.14.");
		}
	};

	TEST_CLASS(SparseVectorTests)
	{
	public:
		TEST_METHOD(TestSparseVectorDot_ZeroVector)
		{
			// Empty sparse vector should have dot product of 0 with itself
			std::unique_ptr<Vector<float, 5>> vec1 = std::make_unique<SparseVector<float, 5>>();
			std::unique_ptr<Vector<float, 5>> vec2 = std::make_unique<SparseVector<float, 5>>();
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(0.0f, dotProduct, L"Dot product of two empty sparse vectors should be 0.");
		}

		TEST_METHOD(TestSparseVectorDot_SingleElement)
		{
			// Sparse vector with single non-zero element
			std::map<size_t, float> data1 = { {0, 5.0f} };
			std::map<size_t, float> data2 = { {0, 3.0f} };
			std::unique_ptr<Vector<float, 5>> vec1 = std::make_unique<SparseVector<float, 5>>(data1);
			std::unique_ptr<Vector<float, 5>> vec2 = std::make_unique<SparseVector<float, 5>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(15.0f, dotProduct, L"Dot product: 5 * 3 = 15.");
		}

		TEST_METHOD(TestSparseVectorDot_NoCommonElements)
		{
			// Sparse vectors with no common non-zero indices
			std::map<size_t, float> data1 = { {0, 2.0f}, {1, 3.0f} };
			std::map<size_t, float> data2 = { {3, 4.0f}, {4, 5.0f} };
			std::unique_ptr<Vector<float, 5>> vec1 = std::make_unique<SparseVector<float, 5>>(data1);
			std::unique_ptr<Vector<float, 5>> vec2 = std::make_unique<SparseVector<float, 5>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(0.0f, dotProduct, L"Dot product with no common elements should be 0.");
		}

		TEST_METHOD(TestSparseVectorDot_PartialOverlap)
		{
			// Sparse vectors with partial overlap: [2, 3, 0, 0, 0] · [0, 4, 0, 5, 0] = 12
			std::map<size_t, float> data1 = { {0, 2.0f}, {1, 3.0f} };
			std::map<size_t, float> data2 = { {1, 4.0f}, {3, 5.0f} };
			std::unique_ptr<Vector<float, 5>> vec1 = std::make_unique<SparseVector<float, 5>>(data1);
			std::unique_ptr<Vector<float, 5>> vec2 = std::make_unique<SparseVector<float, 5>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(12.0f, dotProduct, L"Dot product: (2*0) + (3*4) + (0*0) + (0*5) + (0*0) = 12.");
		}

		TEST_METHOD(TestSparseVectorDot_AllCommonElements)
		{
			// Sparse vectors with all elements at same indices: [1, 2, 3, 0, 0] · [2, 3, 4, 0, 0] = 2 + 6 + 12 = 20
			std::map<size_t, float> data1 = { {0, 1.0f}, {1, 2.0f}, {2, 3.0f} };
			std::map<size_t, float> data2 = { {0, 2.0f}, {1, 3.0f}, {2, 4.0f} };
			std::unique_ptr<Vector<float, 5>> vec1 = std::make_unique<SparseVector<float, 5>>(data1);
			std::unique_ptr<Vector<float, 5>> vec2 = std::make_unique<SparseVector<float, 5>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(20.0f, dotProduct, L"Dot product: (1*2) + (2*3) + (3*4) = 20.");
		}

		TEST_METHOD(TestSparseVectorDot_WithScalarVector)
		{
			// Sparse vector dot product with scalar vector
			std::map<size_t, float> sparseData = { {0, 1.0f}, {2, 2.0f} };
			std::array<float, 5> scalarData = { 1.0f, 0.0f, 2.0f, 0.0f, 0.0f };
			std::unique_ptr<Vector<float, 5>> sparseVec = std::make_unique<SparseVector<float, 5>>(sparseData);
			std::unique_ptr<Vector<float, 5>> scalarVec = std::make_unique<ScalarVector<float, 5>>(scalarData);
			float dotProduct = sparseVec->Dot(*scalarVec);
			Assert::AreEqual(5.0f, dotProduct, L"Dot product: (1*1) + (0*0) + (2*2) + (0*0) + (0*0) = 5.");
		}

		TEST_METHOD(TestSparseVectorDot_NegativeValues)
		{
			// Sparse vector with negative values: [-1, 2, 0, 0, 0] · [3, -1, 0, 0, 0] = -3 - 2 = -5
			std::map<size_t, float> data1 = { {0, -1.0f}, {1, 2.0f} };
			std::map<size_t, float> data2 = { {0, 3.0f}, {1, -1.0f} };
			std::unique_ptr<Vector<float, 5>> vec1 = std::make_unique<SparseVector<float, 5>>(data1);
			std::unique_ptr<Vector<float, 5>> vec2 = std::make_unique<SparseVector<float, 5>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(-5.0f, dotProduct, L"Dot product: (-1*3) + (2*-1) = -5.");
		}

		TEST_METHOD(TestSparseVectorDot_IntegerType)
		{
			// Test sparse vector with integer type
			std::map<size_t, int> data1 = { {0, 2}, {2, 3} };
			std::map<size_t, int> data2 = { {0, 4}, {2, 2} };
			std::unique_ptr<Vector<int, 5>> vec1 = std::make_unique<SparseVector<int, 5>>(data1);
			std::unique_ptr<Vector<int, 5>> vec2 = std::make_unique<SparseVector<int, 5>>(data2);
			float dotProduct = vec1->Dot(*vec2);
			Assert::AreEqual(14.0f, dotProduct, L"Dot product with int type: (2*4) + (3*2) = 14.");
		}

		TEST_METHOD(TestSparseVectorAccessor_MissingElement)
		{
			// Accessing missing elements should return 0
			std::map<size_t, float> data = { {0, 5.0f}, {3, 2.0f} };
			auto vec = std::make_unique<SparseVector<float, 5>>(data);
			Assert::AreEqual(5.0f, (*vec)[0], L"Element at index 0 should be 5.");
			Assert::AreEqual(0.0f, (*vec)[1], L"Missing element at index 1 should return 0.");
			Assert::AreEqual(2.0f, (*vec)[3], L"Element at index 3 should be 2.");
		}
	};
}
