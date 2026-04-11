#include "pch.h"
#include "Vector.h"

template <FloatOrInt T, size_t N>
Vector<T, N>::Vector()
{
    m_data.fill(T{});  // Initialize all elements to the default value of T
}

template <FloatOrInt T, size_t N>
float Vector<T, N>::Dot(const Vector<T, N>& other) const
{
    float result = 0.0f;
    for (size_t i = 0; i < N; ++i)
    {
        // Perform the dot product calculation
        result += m_data[i] * other.m_data[i];
    }
    return result;
}

