#include "scale_forward.h"


Eigen::MatrixXd scaled_backward(const std::vector<std::vector<double>>& O, const Eigen::MatrixXd& T, const std::vector<Eigen::VectorXd>& u, const std::vector<Eigen::MatrixXd>& cov, std::vector<double>& pi, const std::vector<int>& S = { 0, 1, 2 }) {
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
            //std::cout <<"beta"  << beta(t,j) << std::endl;
        }
        //std::cout << "G" << beta.size() << std::endl;
        //std::cout << "G" << beta.row(0) << std::endl;
        G[t] = beta.row(t).sum();
        //std::cout << "G" << G << std::endl;
        for (size_t j = 0; j < T.rows(); j++) {
            beta_hat(t, j) = beta(t, j) / G[t];
        } 
    }
    //std::cout << "G" << G << std::endl;
    //std::cout << "BETA" << beta << std::endl;
    //std::cout << "BETA_HAT" << beta_hat << std::endl;

    return beta_hat;
}
