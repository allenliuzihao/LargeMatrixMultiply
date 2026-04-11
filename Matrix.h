#pragma once

// matrix of M vectors of size N, with elements of type T.
//  The columnMajor flag indicates whether the data is stored in column-major order (true) or row-major order (false).
template <typename VectorType, size_t M, size_t N>
class Matrix
{
public:
    // VectorType is either ScalarVector<T, N> or SparseVector<T, N>
    using ElementType = typename VectorType::element_type;  // Extract T from VectorType
    using ResultVectorType = typename VectorType::template Rebind<M>;

    ResultVectorType operator*(const VectorType& vec) const
    {
        ResultVectorType result;
        for (size_t i = 0; i < M; ++i)
        {
            result[i] = m_data[i].Dot(vec);
        }
        return result;
    }

    ElementType operator()(size_t row, size_t col) const
    {
        if (this->columnMajor)
        {
            return m_data[col][row]; // Access column-major
        }
        else
        {
            return m_data[row][col]; // Access row-major
        }
    }

    void SetColumnMajor(bool value)
    {
        this->columnMajor = value;
    }

    void SetVector(size_t i, const VectorType& v)
    {
        m_data[i] = v;
    }

    inline size_t GetNumVectors() const { return M; }
    inline size_t GetVectorSize() const { return N; }

    // Transpose operator: swaps rows and columns
    // Returns a Matrix<T, N, M> (dimensions swapped)
    virtual std::unique_ptr<Matrix<ResultVectorType, N, M>> FlipStorageFormat() const = 0;

protected:
    bool columnMajor = false; // default to row major.
    std::array<VectorType, M> m_data{};  // ✅ Now stores concrete type

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
        result.newColumnMajor = !this->columnMajor;

        std::vector<ElementType> vData;
        vData.reserve(M);

        if (this->columnMajor)
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
    using Base = Matrix<ScalarVector<T, N>, M, N>;
    using ResultVectorType = typename Base::ResultVectorType;  // ✅ CRITICAL

    std::unique_ptr<Matrix<ResultVectorType, N, M>> FlipStorageFormat() const override
    {
        // ✅ Use common ComputeTranspose() from base class
        auto transposed = this->FlipStorageFormatHelper();

        std::unique_ptr<ScalarMatrix<T, N, M>> result = std::make_unique<ScalarMatrix<T, N, M>>();
        result->SetColumnMajor(transposed.newColumnMajor);

        // ✅ Create ScalarVectors from transposed data
        for (size_t i = 0; i < transposed.vectorData.size(); ++i)
        {
            result->SetVector(i, ResultVectorType(transposed.vectorData[i]));
        }
        return result;
    }

};

template <typename T, size_t M, size_t N>
class SparseMatrix : public Matrix<SparseVector<T, N>, M, N>
{
public: 
    using Base = Matrix<SparseVector<T, N>, M, N>;
    using ResultVectorType = typename Base::ResultVectorType; 

    std::unique_ptr<Matrix<ResultVectorType, N, M>> FlipStorageFormat() const override
    {
        // ✅ Use common ComputeTranspose() from base class
        auto transposed = this->FlipStorageFormatHelper();

        std::unique_ptr<SparseMatrix<T, N, M>> result = std::make_unique<SparseMatrix<T, N, M>>();
        result->SetColumnMajor(transposed.newColumnMajor);

        // ✅ Create SparseVectors from transposed data
        for (size_t i = 0; i < transposed.vectorData.size(); ++i)
        {
            result->SetVector(i, ResultVectorType(transposed.vectorData[i]));
        }
        return result;
    }
};
