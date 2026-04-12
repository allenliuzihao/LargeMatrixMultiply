#pragma once

template <FloatOrInt T, size_t N>
class Vector
{
public:
    using element_type = T;  

    virtual float Dot(const Vector<T, N>& other) const = 0;
    virtual const T& operator[](size_t index) const = 0;

    virtual size_t TotalMemoryBytes() const = 0;
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

    // For std::array and other fixed-size containers (compile-time safety)
    explicit ScalarVector(std::span<const T, N> data)
    {
        std::copy(data.begin(), data.end(), m_data.begin());
    }

    // For std::vector (runtime check, more flexible)
    template <size_t Extent = std::dynamic_extent>
    explicit ScalarVector(std::span<const T> data) requires (Extent == std::dynamic_extent)
    {
        assert(data.size() == N);
        std::copy(data.begin(), data.end(), m_data.begin());
    }

    // Protected getter to access elements
    const T& operator[](size_t index) const override
    {
        assert(index < m_data.size());
        return m_data[index];
    }

    T& operator[](size_t index) 
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

    size_t TotalMemoryBytes() const override
    {
        return sizeof(m_data[0]) * m_data.size();
    }

private:
    std::array<T, N> m_data;  // Array of N elements of type T
};

template <FloatOrInt T, size_t N>
class SparseVector : public Vector<T, N>
{
public:
    class SparseElementProxy
    {
    public:
        SparseElementProxy(SparseVector& vec, size_t index)
            : vec(vec), index(index)
        {
        }

        operator T() const { return static_cast<const SparseVector&>(vec)[index]; }

        SparseElementProxy& operator=(const T& value)
        {
            vec.SetValue(index, value);
            return *this;
        }

    private:
        SparseVector& vec;
        size_t index;
    };

    // Explicit default constructor
    SparseVector()
    {
        m_indices.clear();
        m_data.clear();
    }

    // For std::array (compile-time safety)
    explicit SparseVector(std::span<const T, N> data)
    {
        InitializeFromSpan(data);
    }

    explicit SparseVector(const ScalarVector<T, N>& scalarVec)
    {
        m_indices.reserve(N);
        m_data.reserve(N);

        for (size_t i = 0; i < N; ++i)
        {
            const T& v = scalarVec[i];
            if (v != T{})
            {
                m_indices.push_back(i);
                m_data.push_back(v);
            }
        }
    }

    // Constructor accepting std::map data
    explicit SparseVector(const std::map<size_t, T>& data)
    {
        m_indices.reserve(data.size());
        m_data.reserve(data.size());

        for (const auto& [index, value] : data)
        {
            assert(index < N);
            if (value != T{})  // Only store non-zero elements
            {
                m_indices.push_back(index);
                m_data.push_back(value);
            }
        }
    }

    // For std::vector (runtime check, more flexible)
    template <size_t Extent = std::dynamic_extent>
    explicit SparseVector(std::span<const T> data) requires (Extent == std::dynamic_extent)
    {
        assert(data.size() == N);
        InitializeFromSpan(data);
    }

    // Get non-zero element count
    size_t GetNonZeroCount() const
    {
        return m_indices.size();
    }

    float Dot(const Vector<T, N>& other) const override
    {
        float result = 0.0f;

        const SparseVector<T, N>* otherSparse = dynamic_cast<const SparseVector<T, N>*>(&other);
        if (otherSparse)
        {
            // Both vectors are sparse, use two-pointer technique
            size_t i = 0, j = 0;
            while (i < m_indices.size() && j < otherSparse->m_indices.size())
            {
                if (m_indices[i] == otherSparse->m_indices[j])
                {
                    result += m_data[i] * otherSparse->m_data[j];
                    ++i;
                    ++j;
                }
                else if (m_indices[i] < otherSparse->m_indices[j])
                {
                    ++i;
                }
                else
                {
                    ++j;
                }
            }
            return result;
        } 

        // One vector is sparse and the other is not, iterate over non-zero elements of the sparse vector
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
        assert(index < N);

        // Binary search on m_indices to find the index
        auto it = std::lower_bound(m_indices.begin(), m_indices.end(), index);
        if (it != m_indices.end() && *it == index)
        {
            return m_data[std::distance(m_indices.begin(), it)];
        }

        // Return default value if not found
        static const T zero = T{};
        return zero;
    }

    SparseElementProxy operator[](size_t index) 
    {
        return SparseElementProxy(*this, index);
    }

    size_t TotalMemoryBytes() const override
    {
        return sizeof(size_t) * m_indices.size() + sizeof(T) * m_data.size();
    }

private:
    template <size_t Extent>
    void InitializeFromSpan(std::span<const T, Extent> data)
    {
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (data[i] != T{})
            {
                m_indices.push_back(i);
                m_data.push_back(data[i]);
            }
        }
    }

    void SetValue(size_t index, const T& value) 
    {
        assert(index < N);
        // Binary search on m_indices to find the index,
        auto it = std::lower_bound(m_indices.begin(), m_indices.end(), index);
        auto pos = std::distance(m_indices.begin(), it);

        if (it != m_indices.end() && *it == index)
        {
            if (value == T{})
            {
                // If the value is zero, we need to remove it from the sparse representation if it exists
                m_indices.erase(it);  // Remove the index
                m_data.erase(m_data.begin() + pos);  // Remove the corresponding value
            }
            else
            {
                m_data[pos] = value; // Update the existing non-zero value
            }
            return;
        }
        
        // If not found, we need to insert a new non-zero element
        m_indices.insert(it, index);  // Insert the new index in sorted order
        m_data.insert(m_data.begin() + pos, value);  // Insert the new value for the new index
    }

    std::vector<size_t> m_indices;
    std::vector<T> m_data;  
};