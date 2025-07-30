#include <iostream>
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using namespace autodiff;
using namespace Eigen;

int main()
{
    // Use the alias provided by autodiff for vector of duals
    VectorXdual x(2);
    x << 2.0, 3.0;

    auto f = [](const VectorXdual &x) -> dual
    {
        return x[0] * x[0] + 4.0 * x[1];
    };

    dual y;
    VectorXd grad(x.size());

    // Note: wrt and at take 'x' by reference;
    // make sure 'x' is the same object here
    gradient(f, wrt(x), at(x), y, grad);

    std::cout << "y = " << y << std::endl;
    std::cout << "grad = " << grad.transpose() << std::endl;

    return 0;
}
