#include "emission.h"


std::pair<Eigen::MatrixXd, Eigen::VectorXd>scaled_forward(const std::vector<std::vector<double>>& O, const Eigen::MatrixXd& T, const std::vector<Eigen::VectorXd>& u, const std::vector<Eigen::MatrixXd>& cov, std::vector<double>& pi, const std::vector<int>& S = { 0, 1, 2 }) {
    Eigen::MatrixXd alpha = Eigen::MatrixXd::Zero(O.size(), T.rows());//O shape (5000,4) 这里O.size()应该是5000
    Eigen::MatrixXd alpha_hat = Eigen::MatrixXd::Zero(O.size(), T.rows());
    Eigen::VectorXd G(O.size());
    //std::cout << O.size() << std::endl;
    //std::cout << T.rows() << std::endl;
    //std::cout << emission(0, O[0], u, cov, S) << std::endl;
    for (size_t j = 0; j < T.rows(); j++) {
        //std::cout << j << std::endl;
        //std::cout << pi[j]  << std::endl;
        //std::cout << emission(j, O[0], u, cov, S) << std::endl;
        alpha(0, j) = pi[j] * emission(j, O[0], u, cov, S); // alpha[0][i] = pi[i] * b_i(Y_0)
    }
    //std::cout <<"alpha"<< alpha << std::endl;
    G[0] = alpha.row(0).sum(); // G[0] = sum(i=1)^(3) alpha[0][i]
    //std::cout <<"G0" << G[0] << std::endl;
    for (size_t j = 0; j < T.rows(); j++) {
        alpha_hat(0, j) = alpha(0, j) / G[0];
    }

    //std::cout << "ALPHA" << alpha<<std::endl;
    for (size_t t = 1; t < O.size(); t++) {
        for (size_t j = 0; j < T.rows(); j++) {
            /*
            double sum_alpha_hat = 0.0;
            for (size_t i = 0; i < T.rows(); i++) {
                sum_alpha_hat += alpha_hat(t - 1,i) * T(i,j);
            }
            */
            //std::cout << "row" << alpha_hat.row(t - 1) << std::endl;
            //std::cout << "col" << T.col(j) << std::endl;
            double sum_alpha_hat = alpha_hat.row(t - 1) * T.col(j);
            //std::cout << sum_alpha_hat << std::endl;
            alpha(t, j) = sum_alpha_hat * emission(j, O[t], u, cov, S); // alpha[t][j] = sum(i=1)^(3) alpha_hat[t-1][i] * a_i_j * b_j(Y_t)
        }
        G[t] = alpha.row(t).sum(); // G[t] = sum(j=1)^(3) alpha[t][j]
        for (size_t j = 0; j < T.rows(); j++) {
            alpha_hat(t, j) = alpha(t, j) / G[t]; // alpha_hat[t][j] = alpha[t][j] / G[t]
        }
    }

    //std::cout << "ALPHA_HAT" << alpha_hat << std::endl;

    return std::make_pair(alpha_hat, G);
}
/*
Eigen::MatrixXd scaled_backward(const std::vector<std::vector<double>>& O, const Eigen::MatrixXd& T, const std::vector<Eigen::VectorXd>& u, const std::vector<Eigen::MatrixXd>& cov, const std::vector<double>& pi, const std::vector<int>& S = { 0, 1, 2 }) {
    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(O.size(), T.rows());//O shape (5000,4) 这里O.size()应该是5000
    Eigen::MatrixXd beta_hat = Eigen::MatrixXd::Zero(O.size(), T.rows());
    Eigen::VectorXd G(O.size());
    //std::cout << O.size() << std::endl;
    //std::cout << T.rows() << std::endl;
    for (size_t j = 0; j < T.rows(); j++) {
        beta(O.size() - 1, j) = 1;
        beta_hat(O.size() - 1, j) = 1;
    }
    // Backward loop from t = T - 2 to 0
    for (int t = static_cast<int>(O.size()) - 2; t >= 0; t--) {
        Eigen::VectorXd list(T.rows());
        for (size_t i = 0; i < T.rows(); i++) {
            list(i) = beta_hat(t + 1, i) * emission(i, O[t + 1], u, cov, S);
        }
        //std::cout << "list" << list << std::endl;
        for (size_t j = 0; j < T.rows(); j++) {
            //std::cout << "T rows" << T.row(j) << std::endl;
            beta(t, j) = T.row(j) * list;
            //std::cout <<"beta" << beta << std::endl;
        }
        //std::cout << "G" << beta.size() << std::endl;
        //std::cout << "G" << beta.row(0) << std::endl;
        G[t] = beta.row(t).sum();
        //std::cout << "G" << G << std::endl;
        for (size_t j = 0; j < T.rows(); j++) {
            beta_hat(t, j) = beta(t, j) / G[t];
        }
        //std::cout << "BETA" << beta << std::endl;
        //std::cout << "BETA_HAT" << beta_hat << std::endl;

        return beta_hat;
    }
}
*/


// Example usage
/*
int main() {
    std::vector<std::vector<double>> O = { {-1,1,1,0},{1,1,-1,0} };
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(3, 3);
    T << 0.8, 0.1, 0.1,
        0.1, 0.8, 0.1,
        0.1, 0.1, 0.8;
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
    cov2 << 2, -0.6, 0.5, 0.8,
        -0.6, 2, 0.7, 0.8,
        0.5, 0.7, 2, 0.8,
        0.8, 0.8, 0.8, 1;
    Eigen::MatrixXd cov3 = Eigen::MatrixXd::Zero(4, 4);
    cov3 << 2, 0.6, 0.5, 0.8,
        0.6, 2, -0.7, 0.8,
        0.5, -0.7, 2, 0.8,
        0.8, 0.8, 0.8, 1;
    std::vector<MatrixXd> cov = { cov1,cov2,cov3 };
    std::vector<double> pi = { 0.3,0.4,0.3 };
    std::vector<int> S = { 0,1,2 };
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> result = scaled_forward(O, T, u, cov, pi, S);
    Eigen::MatrixXd alpha_hat = result.first;
    Eigen::VectorXd G = result.second;
    // Print the alpha_hat matrix
    std::cout << "Alpha Hat Matrix:" << std::endl;
    std::cout << alpha_hat << std::endl;
    std::cout << "G " << G << std::endl;

    return 0;
}
*/
