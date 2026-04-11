#pragma once

// matrix of M vectors of size N, with elements of type T.
//  The columnMajor flag indicates whether the data is stored in column-major order (true) or row-major order (false).
template <FloatOrInt T, size_t M, size_t N>
class Matrix
{
public:
    Matrix(bool isColumnMajor = false) : m_isColumnMajor(isColumnMajor) {}
    
    inline size_t GetNumVectors() const { return M; }
    inline size_t GetVectorSize() const { return N; }

    virtual std::unique_ptr<ScalarVector<T, M>> Multiply(const ScalarVector<T, N>& vec) const = 0;

protected:
    bool m_isColumnMajor = false;  // Default to row-major order
};

template <FloatOrInt T, size_t M, size_t N>
class ScalarMatrix : public Matrix<T, M, N>
{
public:
    std::unique_ptr<ScalarVector<T, M>> Multiply(const ScalarVector<T, N>& vec) const override
    {

    }
private:

};

template <FloatOrInt T, size_t M, size_t N>
class SparseMatrix : public Matrix<T, M, N>
{
public: 
    SparseMatrix() : Matrix<T, M, N>()
    {
        m_values.clear();
        m_pointers.clear();
        m_indices.clear();
    }

    SparseMatrix(const T* data, size_t size, bool isColumnMajor = false) : Matrix<T, M, N>(isColumnMajor)
    {
        assert(size == M * N);
        if (isColumnMajor)
        {
            BuildCSCMatrix(data, size);
        }
        else
        {
            BuildCSRMatrix(data, size);
        }
    }

    void FlipStorageFormat()
    {
        // given the current storage format, we can convert it to the other format
        if (this->m_isColumnMajor)
        {
            ConvertCSCtoCSR();
        }
        else
        {
            ConvertCSRtoCSC();
        }
        this->m_isColumnMajor = !this->m_isColumnMajor; // Toggle the storage format flag
    }

    std::unique_ptr<ScalarVector<T, M>> Multiply(const ScalarVector<T, N>& vec) const override
    {
        return nullptr;
    }

private:
    void BuildCSRMatrix(const T* data, size_t size)
    {
        // Implementation for building a CSR (Compressed Sparse Row) matrix from the input data
        // iterate each row
        for (size_t i = 0; i < M; ++i)
        {
            m_pointers.push_back(m_values.size()); // Start of the current row in values and indices
            for (size_t j = 0; j < N; ++j)
            {
                T value = data[i * N + j]; // Accessing the element in row-major order
                if (value != T{}) // Assuming T{} is the default value representing zero
                {
                    m_values.push_back(value); // Store non-zero value
                    m_indices.push_back(j);    // Store column index of the non-zero value
                }
            }
        }
        m_pointers.push_back(m_values.size()); // End of the last row
    }

    void BuildCSCMatrix(const T* data, size_t size)
    {
        // Implementation for building a CSC (Compressed Sparse Column) matrix from the input data
        // iterate each column
        for (size_t j = 0; j < N; ++j)
        {
            m_pointers.push_back(m_values.size()); // Start of the current column in values and indices
            for (size_t i = 0; i < M; ++i)
            {
                T value = data[i * N + j]; // Accessing the element in column-major order
                if (value != T{}) // Assuming T{} is the default value representing zero
                {
                    m_values.push_back(value); // Store non-zero value
                    m_indices.push_back(i);    // Store row index of the non-zero value
                }
            }
        }
        m_pointers.push_back(m_values.size()); // End of the last column
    }

    void ConvertCSRtoCSC()
    {
        // Implementation for converting CSR format to CSC format
        std::vector<uint32_t> colPrefix(N, 0); // Count of non-zero entries in each column
        for (size_t i = 0; i < M; ++i)
        {
            // Iterate through the non-zero entries in the current row
            for (size_t j = m_pointers[i]; j < m_pointers[i + 1]; ++j)
            {
                colPrefix[m_indices[j]]++;
            }
        }

        // compute prefix sum to get column pointers
        uint32_t cumulativeCount = 0;
        for (size_t j = 0; j < N; ++j)
        {
            uint32_t temp = colPrefix[j];
            colPrefix[j] = cumulativeCount;
            cumulativeCount += temp;
        }
        colPrefix.push_back(cumulativeCount);

        // Create new vectors for the transposed matrix
        std::vector<uint32_t> newPointers = colPrefix; // Copy the column pointers to a new vector
        size_t numValues = m_values.size();
        std::vector<T> newValues(numValues); // Create a new vector for values
        std::vector<uint32_t> newIndices(numValues); // Create a new vector for indices

        // i is row index, j is column index
        for (size_t r = 0; r < M; ++r)
        {
            // Iterate through the non-zero entries in the current row
            for (size_t c = m_pointers[r]; c < m_pointers[r + 1]; ++c)
            {
                uint32_t colIndex = m_indices[c];
                uint32_t destIndex = colPrefix[colIndex]++; // Get the next available position in the column

                newIndices[destIndex] = r; // Store the row index in the indices array
                // Move the value to the correct position in the values array
                newValues[destIndex] = m_values[c];
            }
        }

        std::swap(m_pointers, newPointers); // Update the pointers to the new column pointers
        std::swap(m_values, newValues);     // Update the values to the new values
        std::swap(m_indices, newIndices);   // Update the indices to the new indices
    }

    void ConvertCSCtoCSR()
    {
        // Implementation for converting CSC format to CSR format

    }

    std::vector<T> m_values{};
    std::vector<T> m_pointers{};
    std::vector<T> m_indices{};
};

