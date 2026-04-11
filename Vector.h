#pragma once

template <FloatOrInt T, size_t N>
class Vector
{
public:
    Vector();
    virtual float Dot(const Vector<T, N>& other) const = 0;
private:
    std::array<T, N> m_data;  // Array of N elements of type T
};