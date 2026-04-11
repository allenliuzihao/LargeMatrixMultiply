#pragma once

template <FloatOrInt T, size_t N>
class Vector
{
public:
    virtual float Dot(const Vector<T, N>& other) const = 0;
    virtual const T& operator[](size_t index) const = 0;
};

template <FloatOrInt T, size_t N>
class ScalarVector : public Vector<T, N>
{
public:
    // Default constructor
    ScalarVector()
    {
        m_data.fill(T{});  // Initialize all elements to the default value of T
    }

    // Constructor accepting array data
    explicit ScalarVector(const std::array<T, N>& data)
    {
        m_data = data;  // Copy the input data to the member variable
    }

    // Protected getter to access elements
    const T& operator[](size_t index) const override
    {
        assert(index < m_data.size());
        return m_data[index];
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
private:
    std::array<T, N> m_data;  // Array of N elements of type T
};

template <FloatOrInt T, size_t N>
class SparseVector : public Vector<T, N>
{
public:
    // Constructor accepting array data
    explicit SparseVector(const std::array<T, N>& data) 
    {
        for (int i = 0; i < N; ++i)
        {
            if (data[i] != T{}) // Only store non-zero elements
            {
                m_indices.push_back(i);
                m_data.push_back(data[i]);
            }
        }
    }

    float Dot(const Vector<T, N>& other) const override
    {
        float result = 0.0f;

        uint32_t i = 0;
        for (size_t index : m_indices)
        {
            // Perform the dot product calculation
            result += this->m_data[i++] * other[index];
        }
        return result;
    }

    // Protected getter to access elements
    const T& operator[](size_t index) const override
    {
        // binary search on m_indices to find the index
        int left = 0, right = m_indices.size() - 1;
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (m_indices[mid] == index)
            {
                return m_data[mid];
            }
            else if (m_indices[mid] < index)
            {
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }
        return T{}; // Return default value if
    }

private:
    std::vector<size_t> m_indices;
    std::vector<T> m_data;  
};