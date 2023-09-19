#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/math/distributions.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Eigen;

double emission(int s, const std::vector<double>& o, const std::vector<VectorXd>& u, const std::vector<MatrixXd>& cov, const std::vector<int>& S) {
    assert(S.size() == u.size() && S.size() == cov.size());

    for (size_t i = 0; i < S.size(); ++i) {
        if (s == S[i]) {
            //std::cout << i << std::endl;
            int dim = 4;
            MatrixXd cov_matrix(dim, dim);
            for (int j = 0; j < dim; ++j) {
                for (int k = 0; k < dim; ++k) {
                    cov_matrix(j, k) = cov[i](j, k);
                }
            }
            VectorXd mean(dim);
            for (int j = 0; j < dim; ++j) {
                mean(j) = u[i](j);
            }
            //std::cout << cov_matrix << std::endl;
            VectorXd x(dim);
            for (int j = 0; j < dim; j++) {
                x(j) = o[j];
            }
            //std::cout << x << std::endl;
            // Compute the Mahalanobis distance
            VectorXd diff = x - mean;
            //std::cout << diff << std::endl;
            // Calculate the determinant
            double det = cov_matrix.determinant();
            //std::cout << det << std::endl;
            // Calculate the inverse
            Eigen::MatrixXd inv = cov_matrix.inverse();
            // Compute the normalization constant
            //std::cout << inv << std::endl;
            double normalization = std::sqrt(std::pow(2.0 * M_PI, cov_matrix.rows()) * det);
            //std::cout << normalization << std::endl;
            double n = diff.transpose() * inv * diff;
            // Compute the PDF
            double pdf = std::exp(-0.5 * n) / normalization;
            //std::cout << pdf << std::endl;
            return pdf;
        }
    }
    return 0.0;
}

// Example usage
/*
int main() {
    std::vector<double> o = { 1, 1, -1, 0 };  // Observation
    Eigen::VectorXd u1(4);
    u1 << 1, 1, -1, 0;
    Eigen::VectorXd u2(4);
    u2 << 1, -1, 1, 0;
    Eigen::VectorXd u3(4);
    u3 << -1, 1, 1, 0;
    std::vector<VectorXd> u = {
        u1,
        u2,
        u3
    };   // Mean
    Eigen::MatrixXd cov1 = Eigen::MatrixXd::Zero(4, 4);
    cov1 << 2, 0.6, -0.5, 0.8,
        0.6, 2, 0.7, 0.8,
        -0.5, 0.7, 2, 0.8,
        0.8, 0.8, 0.8, 1;
    Eigen::MatrixXd cov2 = Eigen::MatrixXd::Zero(4, 4);
    cov2 << 2, 0.6, -0.5, 0.8,
        0.6, 2, 0.7, 0.8,
        -0.5, 0.7, 2, 0.8,
        0.8, 0.8, 0.8, 1;
    Eigen::MatrixXd cov3 = Eigen::MatrixXd::Zero(4, 4);
    cov3 << 2, 0.6, -0.5, 0.8,
        0.6, 2, 0.7, 0.8,
        -0.5, 0.7, 2, 0.8,
        0.8, 0.8, 0.8, 1;
    std::vector<MatrixXd> cov = { cov1,cov2,cov3};
    std::vector<int> S = { 0,1,2 };  // State space

    int s = 0;  // Hidden state

    double result = emission(s, o, u, cov, S);

    std::cout << "Emission probability: " << result << std::endl;

    return result;
}
*/
