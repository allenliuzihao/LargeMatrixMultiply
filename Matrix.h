#pragma once

// matrix of M vectors of size N, with elements of type T.
//  The columnMajor flag indicates whether the data is stored in column-major order (true) or row-major order (false).
template <typename T, size_t M, size_t N>
class Matrix
{
public:
    Vector<T, M> operator*(const Vector<T, N>& vec) const
    {
        // Implement matrix-vector multiplication
        Vector<T, M> result; // Result will be a vector of size M
        for (size_t i = 0; i < M; ++i)
        {
            result[i] = m_data[i].Dot(vec); // Dot product of each row (or column) with the input vector
        }
        return result;
    }

    // Transpose operator: swaps rows and columns
    // Returns a Matrix<T, N, M> (dimensions swapped)
    virtual Matrix<T, N, M> FlipStorageFormat() const = 0;

protected:
    bool columnMajor = false; // default to row major.
    std::vector<Vector<T, N>> m_data{};
};

template <typename T, size_t M, size_t N>
class ScalarMatrix : public Matrix<T, M, N>
{
public:
    Matrix<T, N, M> FlipStorageFormat() const override
    {
        Matrix<T, N, M> result;
        result.columnMajor = !this->columnMajor;  // Flip storage format

        std::vector<T> vData{};
        vData.resize(M); // Pre-allocate for M elements in each row

        if (this->columnMajor)
        {
            // Current: Column-major (M columns of size N)
            // Result: Row-major (N rows of size M)
            // m_data[i] is column i
            // Extract row j: element j from each column
            for (size_t row = 0; row < N; ++row)
            {
                vData.clear();
                for (size_t col = 0; col < M; ++col)
                {
                    vData.push_back(this->m_data[col][row]);
                }
                result.m_data.push_back(ScalarVector<T, M>(vData));
            }
        }
        else
        {
            // Current: Row-major (M rows of size N)
            // Result: Column-major (N columns of size M)
            // m_data[i] is row i
            // Extract column j: element j from each row
            for (size_t col = 0; col < N; ++col)
            {
                vData.clear();
                for (size_t row = 0; row < M; ++row)
                {
                    vData.push_back(this->m_data[row][col]);
                }
                result.m_data.push_back(ScalarVector<T, M>(vData));
            }
        }
        return result;
    }

};

template <typename T, size_t M, size_t N>
class SparseMatrix : public Matrix<T, M, N>
{
public: 
    Matrix<T, N, M> FlipStorageFormat() const override
    {

    }
};
