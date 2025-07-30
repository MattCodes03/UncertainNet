#include <iostream>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "Core/Tensor.h"

using autodiff::dual;
using autodiff::gradient;

int main()
{
    Tensor<dual> t1(2, 2);
    Tensor<dual> t2(2, 2);

    // Initialize t1
    t1(0, 0) = 1.0;
    t1(0, 1) = 2.0;
    t1(1, 0) = 3.0;
    t1(1, 1) = 4.0;

    // Initialize t2
    t2(0, 0) = 5.0;
    t2(0, 1) = 6.0;
    t2(1, 0) = 7.0;
    t2(1, 1) = 8.0;

    // Matrix multiply
    auto t3 = t1.matmul(t2);

    // Matrix Transpose
    auto t4 = t2.transpose();

    // Print t3
    std::cout << "t3 = \n";
    for (int i = 0; i < t3.rows(); ++i)
    {
        for (int j = 0; j < t3.cols(); ++j)
            std::cout << t3(i, j) << " ";
        std::cout << "\n";
    }

    // Print t4
    std::cout << "t4 = \n";
    for (int i = 0; i < t4.rows(); ++i)
    {
        for (int j = 0; j < t4.cols(); ++j)
            std::cout << t4(i, j) << " ";
        std::cout << "\n";
    }

    // Now, define a scalar function of the elements of t1 (flattened)
    auto f = [](const Eigen::Matrix<dual, Eigen::Dynamic, 1> &x)
    {
        // Let's define a simple quadratic function: sum of squares
        dual s = 0.0;
        for (int i = 0; i < x.size(); ++i)
            s += x(i) * x(i);
        return s;
    };

    // Flatten t1's data to Eigen vector of dual for gradient calculation
    Eigen::Matrix<dual, Eigen::Dynamic, 1> t1_vec(t1.rows() * t1.cols());
    for (int i = 0; i < t1.rows(); ++i)
        for (int j = 0; j < t1.cols(); ++j)
            t1_vec(i * t1.cols() + j) = t1(i, j);

    dual y;
    Eigen::VectorXd grad(t1_vec.size());

    // Compute gradient of f at t1_vec
    gradient(f, wrt(t1_vec), at(t1_vec), y, grad);

    std::cout << "\nFunction value (sum of squares) = " << y << "\n";
    std::cout << "Gradient = ";
    for (int i = 0; i < grad.size(); ++i)
        std::cout << grad(i) << " ";
    std::cout << "\n";

    return 0;
}
