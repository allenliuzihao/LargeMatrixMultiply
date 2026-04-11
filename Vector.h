#pragma once

template <FloatOrInt T, size_t N>
class Vector
{
protected:
    Vector()
    {
        m_data.fill(T{});  // Initialize all elements to the default value of T
    }

    Vector(const std::array<T, N>& data)
    {
        m_data = data;  // Copy the input data to the member variable
    }


    std::array<T, N> m_data;  // Array of N elements of type T
public:
    virtual float Dot(const Vector<T, N>& other) const = 0;

    // Protected getter to access elements
    const T& operator[](size_t index) const
    {
        assert(index < m_data.size());
        return m_data[index];
    }
};

template <FloatOrInt T, size_t N>
class ScalarVector : public Vector<T, N>
{
public:
    // Default constructor
    ScalarVector() : Vector<T, N>()
    {
    }

    // Constructor accepting array data
    explicit ScalarVector(const std::array<T, N>& data) : Vector<T, N>(data)
    {
    }

    float Dot(const Vector<T, N>& other) const override
    {
        float result = 0.0f;
        for (size_t i = 0; i < N; ++i)
        {
            // Perform the dot product calculation
            result += this->m_data[i] * other[i];
        }
        return result;
    }
};