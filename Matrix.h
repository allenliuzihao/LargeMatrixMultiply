#pragma once

// matrix of M vectors of size N, with elements of type T.
//  The columnMajor flag indicates whether the data is stored in column-major order (true) or row-major order (false).
template <typename VectorType, size_t M, size_t N>
class Matrix
{
public:
    // VectorType is either ScalarVector<T, N> or SparseVector<T, N>
    using ElementType = typename VectorType::element_type;  // Extract T from VectorType

    VectorType operator*(const VectorType& vec) const
    {
        VectorType result;
        for (size_t i = 0; i < M; ++i)
        {
            result[i] = m_data[i].Dot(vec);
        }
        return result;
    }

    // Transpose operator: swaps rows and columns
    // Returns a Matrix<T, N, M> (dimensions swapped)
    virtual Matrix<VectorType, N, M> FlipStorageFormat() const = 0;

protected:
    bool columnMajor = false; // default to row major.
    std::vector<VectorType> m_data{};  // ✅ Now stores concrete type

    // Common transpose data structure
    struct MatrixRawData
    {
        bool newColumnMajor;
        std::vector<std::vector<ElementType>> vectorData;
    };

    // Unified transpose logic - in base class
    MatrixRawData FlipStorageFormatHelper() const
    {
        MatrixRawData result;
        result.newColumnMajor = !columnMajor;

        std::vector<ElementType> vData;
        vData.reserve(M);

        if (columnMajor)
        {
            // Column-major -> Row-major (M columns of size N become N rows of size M)
            for (size_t row = 0; row < N; ++row)
            {
                vData.clear();
                for (size_t col = 0; col < M; ++col)
                {
                    vData.push_back(m_data[col][row]);
                }
                result.vectorData.push_back(vData);
            }
        }
        else
        {
            // Row-major -> Column-major (M rows of size N become N columns of size M)
            for (size_t col = 0; col < N; ++col)
            {
                vData.clear();
                for (size_t row = 0; row < M; ++row)
                {
                    vData.push_back(m_data[row][col]);
                }
                result.vectorData.push_back(vData);
            }
        }
        return result;
    }
};

template <typename T, size_t M, size_t N>
class ScalarMatrix : public Matrix<ScalarVector<T, N>, M, N>
{
public:
    using VectorType = ScalarVector<T, M>;

    Matrix<VectorType, N, M> FlipStorageFormat() const override
    {
        // ✅ Use common ComputeTranspose() from base class
        auto transposed = this->FlipStorageFormatHelper();

        ScalarMatrix<T, N, M> result;
        result.columnMajor = transposed.newColumnMajor;

        // ✅ Create ScalarVectors from transposed data
        for (const auto& vectorData : transposed.vectorData)
        {
            result.m_data.push_back(VectorType(vectorData));
        }
        return result;
    }

};

template <typename T, size_t M, size_t N>
class SparseMatrix : public Matrix<SparseVector<T, N>, M, N>
{
public: 
    using VectorType = SparseVector<T, M>;

    Matrix<VectorType, N, M> FlipStorageFormat() const override
    {
        // ✅ Use common ComputeTranspose() from base class
        auto transposed = this->FlipStorageFormatHelper();

        SparseMatrix<T, N, M> result;
        result.columnMajor = transposed.newColumnMajor;

        // ✅ Create SparseVectors from transposed data
        for (const auto& vectorData : transposed.vectorData)
        {
            result.m_data.push_back(VectorType(vectorData));
        }

        return result;
    }
};
