#pragma once

// matrix of M vectors of size N, with elements of type T.
//  The columnMajor flag indicates whether the data is stored in column-major order (true) or row-major order (false).
template <FloatOrInt T, size_t M, size_t N>
class Matrix
{
public:
    Matrix(bool isColumnMajor = false) : m_isColumnMajor(isColumnMajor) {}
    
    inline size_t GetNumRows() const { return M; }
    inline size_t GetNumCols() const { return N; }
    inline bool IsColumnMajor() const { return m_isColumnMajor; }


    virtual void FlipStorageFormat() = 0;
    virtual std::unique_ptr<ScalarVector<T, N>> LeftMultiply(const ScalarVector<T, M>& vec) const = 0;
    virtual std::unique_ptr<ScalarVector<T, M>> RightMultiply(const ScalarVector<T, N>& vec) const = 0;

    virtual size_t TotalMemoryBytes() const = 0;
protected:
    bool m_isColumnMajor = false;  // Default to row-major order
};

template <FloatOrInt T, size_t M, size_t N>
class ScalarMatrix : public Matrix<T, M, N>
{
public:
    ScalarMatrix(bool isColumnMajor = false) : Matrix<T, M, N>(isColumnMajor)
    {
        m_size = M * N;
        m_data = new T[m_size](); // Allocate memory for M*N elements and initialize to default value of T
        std::memset(m_data, 0, m_size * sizeof(T)); // Initialize all elements to zero (or default value of T)
    }

    ~ScalarMatrix()
    {
        delete[] m_data; // Free the allocated memory
    }

    void FlipStorageFormat() override
    {
        // given the current storage format, we can transpose the matrix to switch between row-major and column-major
        T* newData = new T[m_size](); // Create a new array for the transposed data
        if (this->m_isColumnMajor)
        {
            for (size_t i = 0; i < N; ++i)
            {
                for (size_t j = 0; j < M; ++j)
                {
                    newData[j * N + i] = m_data[i * M + j]; // Transpose the element
                }
            }
        }
        else
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    newData[j * M + i] = m_data[i * N + j]; // Transpose the element
                }
            }
        }

        std::swap(m_data, newData); // Update the data with the transposed data
        delete[] newData; // Free the temporary array
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
                sum += m_data[i * N + j] * vec[j];
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
                sum += m_data[j * M + i] * vec[i];
            }
            (*result)[j] = sum;
        }
        return result;
    }

    template <size_t K>
    std::unique_ptr<ScalarMatrix<T, M, K>> RightMultiply(const ScalarMatrix<T, N, K>& other) const
    {
        return nullptr;
    }

    size_t TotalMemoryBytes() const override
    {
        return sizeof(T) * m_size; // Memory used by the matrix data
    }

private:
    // raw matrix data in row major format.
    T* m_data = nullptr;
    size_t m_size;
};

template <FloatOrInt T, size_t M, size_t N>
class SparseMatrix : public Matrix<T, M, N>
{
public: 
    SparseMatrix(bool isColumnMajor = false) : Matrix<T, M, N>(isColumnMajor)
    {
        m_values.clear();
        m_pointers.clear();
        m_indices.clear();
    }

    // is the data in column major order or row major order, we can build the matrix accordingly
    SparseMatrix(const T* data, size_t size, bool isColumnMajor = false) : Matrix<T, M, N>(isColumnMajor)
    {
        assert(size == M * N);

        m_values.clear();
        m_indices.clear();
        m_pointers.clear();

        if (isColumnMajor)
        {
            m_pointers.reserve(N + 1);
            BuildCSCMatrix(data, size);
        }
        else
        {
            m_pointers.reserve(M + 1);
            BuildCSRMatrix(data, size);
        }
    }

    SparseMatrix(const ScalarMatrix<T, M, N>& matrix, bool isColumnMajor = false) : Matrix<T, M, N>(isColumnMajor)
    {
        m_values.clear();
        m_indices.clear();
        m_pointers.clear();

        if (isColumnMajor)
        {
            m_pointers.reserve(N + 1);
            if (matrix.isColumnMajor())
            {
                BuildCSCMatrixColumnMajor(matrix.m_data, M * N); // Assuming m_data is a 1D array storing the matrix data in column-major order
            }
            else
            {
                BuildCSCMatrixRowMajor(matrix.m_data, M * N); // Assuming m_data is a 1D array storing the matrix data in row-major order
            }
        }
        else
        {
            m_pointers.reserve(M + 1);
            BuildCSRMatrix(matrix.m_data, M * N); // Assuming m_data is a 1D array storing the matrix data
        }

        // Set the storage format based on the input matrix
        this->m_isColumnMajor = isColumnMajor; 
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

    inline size_t GetNonZeroCount() const { return m_values.size(); }

    template <size_t K>
    std::unique_ptr<ScalarMatrix<T, M, K>> RightMultiply(const SparseMatrix<T, N, K>& other) const
    {
        return nullptr;
    }

    size_t TotalMemoryBytes() const override
    {
        return sizeof(T) * m_values.size() + sizeof(uint32_t) * m_indices.size() + sizeof(uint32_t) * m_pointers.size(); // Memory used by the sparse matrix data
    }

private:
    // data is stored in row major order, so we can iterate through each row and build the CSR format
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

    void BuildCSCMatrixColumnMajor(const T* data, size_t size)
    {
        // Implementation for building a CSR (Compressed Sparse Row) matrix from the input data
        for (size_t c = 0; c < N; ++c)
        {
            m_pointers.push_back(m_values.size()); // Start of the current row in values and indices
            for (size_t r = 0; r < M; ++r)
            {
                T value = data[c * M + r]; // Accessing the element in column-major order
                if (value != T{}) // Assuming T{} is the default value representing zero
                {
                    m_values.push_back(value); 
                    m_indices.push_back(static_cast<uint32_t>(r));    
                }
            }
        }
        m_pointers.push_back(static_cast<uint32_t>(m_values.size())); // End of the last row
    }

    // this assumes data is stored in row major order, so we build CSC matrix.
    void BuildCSCMatrixRowMajor(const T* data, size_t size)
    {
        m_pointers.resize(N + 1, 0); // Initialize column pointers with size N+1

        // Implementation for building a CSC (Compressed Sparse Column) matrix from the input data
        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                T value = data[i * N + j]; // Accessing the element in row-major order
                if (value != T{}) // Assuming T{} is the default value representing zero
                {
                    m_pointers[j + 1]++; // Increment the count of non-zero entries in the current column
                }
            }
        }

        // Compute prefix sum to get column pointers
        for (size_t j = 1; j <= N; ++j)
        {
            m_pointers[j] += m_pointers[j - 1];
        }

        size_t nnz = m_pointers[N];
        m_indices.resize(nnz);
        m_values.resize(nnz);
        std::vector<uint32_t> currentPosition = m_pointers; // Create a copy of column pointers to track current position in each column

        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                T value = data[i * N + j]; // Accessing the element in row-major order
                if (value != T{}) // Assuming T{} is the default value representing zero
                {
                    uint32_t destIndex = currentPosition[j]++; // Get the next available position in the column
                    m_indices[destIndex] = static_cast<uint32_t>(i); // Store row index of the non-zero value
                    m_values[destIndex] = value; // Store the non-zero value
                }
            }
        }
    }

    void ConvertCSRtoCSC()
    {
        // Implementation for converting CSR format to CSC format
        std::vector<uint32_t> colPrefix(N + 1, 0); // Count of non-zero entries in each column
        for (size_t i = 0; i < M; ++i)
        {
            // Iterate through the non-zero entries in the current row
            for (size_t j = m_pointers[i]; j < m_pointers[i + 1]; ++j)
            {
                colPrefix[m_indices[j] + 1]++;
            }
        }

        // compute prefix sum to get column pointers
        for (size_t j = 0; j < N; ++j)
        {
            colPrefix[j + 1] += colPrefix[j];
        }

        // Create new vectors for the transposed matrix
        std::vector<uint32_t> newPointers = colPrefix; // Copy the column pointers to a new vector
        size_t numValues = colPrefix[N];
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
        std::vector<uint32_t> rowPrefix(M + 1, 0); // Count of non-zero entries in each row
        // for each column, iterate through the non-zero entries and count how many entries are in each row
        for (size_t i = 0; i < N; ++i)
        {
            // Iterate through the non-zero entries in the current column
            for (size_t j = m_pointers[i]; j < m_pointers[i + 1]; ++j)
            {
                rowPrefix[m_indices[j] + 1]++;
            }
        }

        // compute prefix sum to get column pointers
        for (size_t i = 0; i < M; ++i)
        {
            rowPrefix[i + 1] += rowPrefix[i];
        }

        // Create new vectors for the transposed matrix
        std::vector<uint32_t> newPointers = rowPrefix; // Copy the row pointers to a new vector
        size_t numValues = rowPrefix[M];
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

