#include "pch.h"
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace VectorTests
{
	TEST_CLASS(VectorTests)
	{
	public:
		TEST_METHOD(TestVectorDot)
		{
            std::unique_ptr<Vector<float, 3>> vec1 = std::make_unique<ScalarVector<float, 3>>();
			std::unique_ptr<Vector<float, 3>> vec2 = std::make_unique<ScalarVector<float, 3>>();
            float dotProduct = vec1->Dot(*vec2);
            Assert::AreEqual(0.0f, dotProduct, L"Dot product of two zero vectors should be 0.");
		}
	};
}
