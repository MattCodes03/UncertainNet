#pragma once

#include <Eigen/Dense>

template <typename Scalar>
class Tensor
{
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Tensor() = default;

    Tensor(int rows, int cols)
        : data_(rows, cols)
    {
        data_.setZero();
    }

    Tensor(const MatrixType &mat)
        : data_(mat)
    {
    }

    // Access element
    Scalar &operator()(int row, int col) { return data_(row, col); }
    const Scalar &operator()(int row, int col) const { return data_(row, col); }

    int rows() const { return data_.rows(); }
    int cols() const { return data_.cols(); }

    // Basic arithmetic operations
    Tensor operator+(const Tensor &other) const { return Tensor(data_ + other.data_); }
    Tensor operator-(const Tensor &other) const { return Tensor(data_ - other.data_); }
    Tensor operator*(const Tensor &other) const { return Tensor(data_.cwiseProduct(other.data_)); }
    Tensor operator*(Scalar scalar) const { return Tensor(data_ * scalar); }

    // Matrix multiplication
    Tensor matmul(const Tensor &other) const { return Tensor(data_ * other.data_); }
    Tensor transpose() const { return Tensor(data_.transpose()); }

    // Get underlying Eigen matrix (const)
    const MatrixType &matrix() const { return data_; }
    MatrixType &matrix() { return data_; }

private:
    MatrixType data_;
};
