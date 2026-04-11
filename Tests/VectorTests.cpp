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
}
