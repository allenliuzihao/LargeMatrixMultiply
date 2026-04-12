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
    inline bool IsColumnMajor() const { return m_isColumnMajor; }


    virtual void FlipStorageFormat() = 0;
    virtual std::unique_ptr<ScalarVector<T, N>> LeftMultiply(const ScalarVector<T, M>& vec) const = 0;
    virtual std::unique_ptr<ScalarVector<T, M>> RightMultiply(const ScalarVector<T, N>& vec) const = 0;

protected:
    bool m_isColumnMajor = false;  // Default to row-major order
};

template <FloatOrInt T, size_t M, size_t N>
class ScalarMatrix : public Matrix<T, M, N>
{
public:
    ScalarMatrix(bool isColumnMajor = false) : Matrix<T, M, N>(isColumnMajor)
    {
        if (isColumnMajor)
        {
            // If column-major, we can store the data in a single vector and calculate indices accordingly
            m_data.resize(N, std::vector<T>(M, T{})); // Initialize a NxM matrix with default values of T
        }
        else
        {
            // If row-major, we can store the data in a single vector and calculate indices accordingly
            m_data.resize(M, std::vector<T>(N, T{})); // Initialize a MxN matrix with default values of T
        }
    }

    void FlipStorageFormat() override
    {
        // given the current storage format, we can transpose the matrix to switch between row-major and column-major
        if (this->m_isColumnMajor)
        {
            // If currently column-major, we need to transpose the data to row-major
            std::vector<std::vector<T>> transposedData(M, std::vector<T>(N)); // Create a new vector for the transposed data
            for (size_t i = 0; i < N; ++i)
            {
                for (size_t j = 0; j < M; ++j)
                {
                    transposedData[j][i] = (*this)(i, j); // Transpose the element
                }
            }
            m_data = std::move(transposedData); // Update the data with the transp
        }
        else
        {
            // If currently row-major, we need to transpose the data to column-major
            std::vector<std::vector<T>> transposedData(N, std::vector<T>(M)); // Create a new vector for the transposed data
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    transposedData[j][i] = (*this)(i, j); // Transpose the element
                }
            }
            m_data = std::move(transposedData); // Update the data with the transposed data
        }
        this->m_isColumnMajor = !this->m_isColumnMajor; // Toggle the storage format flag
    }

    std::unique_ptr<ScalarVector<T, M>> RightMultiply(const ScalarVector<T, N>& vec) const override
    {
        if (this->m_isColumnMajor)
        {
            throw std::runtime_error("Right multiplication is not supported for column-major format. Please flip the storage format to row-major before performing right multiplication.");
        }

        std::unique_ptr<ScalarVector<T, M>> result = std::make_unique<ScalarVector<T, M>>(); // Create a result vector of size M
        // for each matrix row, we can compute the dot product of the row with the input vector
        for (size_t i = 0; i < M; ++i)
        {
            T sum = T{};
            for (size_t j = 0; j < N; ++j)
            {
                sum += (*this)(i, j) * vec[j];
            }
            (*result)[i] = sum;
        }
        return result;
    }

    std::unique_ptr<ScalarVector<T, N>> LeftMultiply(const ScalarVector<T, M>& vec) const
    {
        if (!this->m_isColumnMajor)
        {
            throw std::runtime_error("Left multiplication is not supported for row-major format. Please flip the storage format to column-major before performing left multiplication.");
        }

        std::unique_ptr<ScalarVector<T, N>> result = std::make_unique<ScalarVector<T, N>>(); // Create a result vector of size N
        for (size_t j = 0; j < N; ++j)
        {
            T sum = T{};
            for (size_t i = 0; i < M; ++i)
            {
                sum += (*this)(i, j) * vec[i];
            }
            (*result)[j] = sum;
        }
        return result;
    }

private:
    const T& operator()(size_t row, size_t col) const
    {
        if (this->m_isColumnMajor)
        {
            return m_data[col][row]; // Accessing as column-major
        }
        return m_data[row][col];
    }

    T& operator()(size_t row, size_t col)
    {
        if (this->m_isColumnMajor)
        {
            return m_data[col][row]; // Accessing as column-major
        }
        return m_data[row][col];
    }

    std::vector<std::vector<T>> m_data{}; // 2D vector to store matrix data
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

    void FlipStorageFormat() override
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

    std::unique_ptr<ScalarVector<T, M>> RightMultiply(const ScalarVector<T, N>& vec) const override
    {
        if (this->m_isColumnMajor)
        {
            throw std::runtime_error("Right multiplication is not supported for column-major format. Please flip the storage format to row-major before performing right multiplication.");
        }

        std::unique_ptr<ScalarVector<T, M>> result = std::make_unique<ScalarVector<T, M>>(); // Create a result vector of size M
        for (size_t i = 0; i < M; ++i)
        {
            // Iterate through the non-zero entries in the current row
            for (size_t j = m_pointers[i]; j < m_pointers[i + 1]; ++j)
            {
                (*result)[i] += m_values[j] * vec[m_indices[j]]; // Multiply the non-zero value with the corresponding vector element and accumulate
            }
        }
        return result;
    }

    std::unique_ptr<ScalarVector<T, N>> LeftMultiply(const ScalarVector<T, M>& vec) const override
    {
        if (!this->m_isColumnMajor)
        {
            throw std::runtime_error("Left multiplication is not supported for row-major format. Please flip the storage format to column-major before performing left multiplication.");
        }

        std::unique_ptr<ScalarVector<T, N>> result = std::make_unique<ScalarVector<T, N>>(); // Create a result vector of size N
        for (size_t j = 0; j < N; ++j)
        {
            // Iterate through the non-zero entries in the current column
            for (size_t i = m_pointers[j]; i < m_pointers[j + 1]; ++i)
            {
                (*result)[j] += m_values[i] * vec[m_indices[i]]; // Multiply the non-zero value with the corresponding vector element and accumulate
            }
        }
        return result;
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
                    m_indices.push_back(static_cast<uint32_t>(j));    // Store column index of the non-zero value
                }
            }
        }
        m_pointers.push_back(static_cast<uint32_t>(m_values.size())); // End of the last row
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
                    m_indices.push_back(static_cast<uint32_t>(i));    // Store row index of the non-zero value
                }
            }
        }
        m_pointers.push_back(static_cast<uint32_t>(m_values.size())); // End of the last column
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
                newIndices[destIndex] = static_cast<uint32_t>(r);
                newValues[destIndex] = m_values[c];
            }
        }

        std::swap(m_values, newValues); // Update the values to the new values
        std::swap(m_indices, newIndices); // Update the indices to the new indices
        std::swap(m_pointers, newPointers); // Update the pointers to the new column pointers
    }

    void ConvertCSCtoCSR()
    {
        // Implementation for converting CSC format to CSR format
        std::vector<uint32_t> rowPrefix(M, 0); // Count of non-zero entries in each row
        // for each column, iterate through the non-zero entries and count how many entries are in each row
        for (size_t i = 0; i < N; ++i)
        {
            // Iterate through the non-zero entries in the current column
            for (size_t j = m_pointers[i]; j < m_pointers[i + 1]; ++j)
            {
                rowPrefix[m_indices[j]]++;
            }
        }

        // compute prefix sum to get column pointers
        uint32_t cumulativeCount = 0;
        for (size_t j = 0; j < M; ++j)
        {
            uint32_t temp = rowPrefix[j];
            rowPrefix[j] = cumulativeCount;
            cumulativeCount += temp;
        }
        rowPrefix.push_back(cumulativeCount);

        // Create new vectors for the transposed matrix
        std::vector<uint32_t> newPointers = rowPrefix; // Copy the row pointers to a new vector
        size_t numValues = m_values.size();
        std::vector<T> newValues(numValues); // Create a new vector for values
        std::vector<uint32_t> newIndices(numValues); // Create a new vector for indices

        // r is row index, c is column index
        for (size_t c = 0; c < N; ++c)
        {
            // Iterate through the non-zero entries in the current column
            for (size_t r = m_pointers[c]; r < m_pointers[c + 1]; ++r)
            {
                uint32_t rowIndex = m_indices[r];
                uint32_t destIndex = rowPrefix[rowIndex]++; // Get the next available position in the row
                newIndices[destIndex] = static_cast<uint32_t>(c);
                newValues[destIndex] = m_values[r];
            }
        }

        std::swap(m_values, newValues); // Update the values to the new values
        std::swap(m_indices, newIndices); // Update the indices to the new indices
        std::swap(m_pointers, newPointers); // Update the pointers to the new column pointers
    }

    std::vector<T> m_values{};
    std::vector<uint32_t> m_pointers{};
    std::vector<uint32_t> m_indices{};
};

