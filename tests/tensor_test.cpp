#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "Core/Tensor.h"

using autodiff::dual;
using autodiff::gradient;

TEST_CASE("Tensor matrix ops and autodiff gradient computation", "[tensor][autodiff]")
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

    // Matrix transpose
    auto t4 = t2.transpose();

    SECTION("Matrix multiply results")
    {
        REQUIRE(t3.rows() == 2);
        REQUIRE(t3.cols() == 2);

        // Expected results of t1 * t2
        // [1*5+2*7, 1*6+2*8]
        // [3*5+4*7, 3*6+4*8]
        REQUIRE(t3(0, 0) == Approx(19.0));
        REQUIRE(t3(0, 1) == Approx(22.0));
        REQUIRE(t3(1, 0) == Approx(43.0));
        REQUIRE(t3(1, 1) == Approx(50.0));
    }

    SECTION("Transpose results")
    {
        REQUIRE(t4.rows() == 2);
        REQUIRE(t4.cols() == 2);

        REQUIRE(t4(0, 0) == Approx(5.0));
        REQUIRE(t4(0, 1) == Approx(7.0));
        REQUIRE(t4(1, 0) == Approx(6.0));
        REQUIRE(t4(1, 1) == Approx(8.0));
    }

    // Define scalar function for autodiff
    auto f = [](const Eigen::Matrix<dual, Eigen::Dynamic, 1> &x)
    {
        dual s = 0.0;
        for (int i = 0; i < x.size(); ++i)
            s += x(i) * x(i);
        return s;
    };

    // Flatten t1's data to Eigen vector for gradient
    Eigen::Matrix<dual, Eigen::Dynamic, 1> t1_vec(t1.rows() * t1.cols());
    for (int i = 0; i < t1.rows(); ++i)
        for (int j = 0; j < t1.cols(); ++j)
            t1_vec(i * t1.cols() + j) = t1(i, j);

    dual y;
    Eigen::VectorXd grad(t1_vec.size());

    SECTION("Gradient computation")
    {
        gradient(f, wrt(t1_vec), at(t1_vec), y, grad);

        // sum of squares = 1^2+2^2+3^2+4^2 = 1+4+9+16 = 30
        REQUIRE(y == Approx(30.0));

        // gradient = 2 * x_i for each element
        REQUIRE(grad.size() == 4);
        REQUIRE(grad(0) == Approx(2.0 * 1.0));
        REQUIRE(grad(1) == Approx(2.0 * 2.0));
        REQUIRE(grad(2) == Approx(2.0 * 3.0));
        REQUIRE(grad(3) == Approx(2.0 * 4.0));
    }
}
