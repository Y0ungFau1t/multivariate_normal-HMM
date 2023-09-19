#include "scale_backward.h"
#include <unsupported/Eigen/CXX11/Tensor>



std::tuple<Eigen::MatrixXd, std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>, std::vector<double>, Eigen::VectorXi>
scaled_baum_welch(const std::vector<std::vector<double>>& O, Eigen::MatrixXd& trans,
    std::vector<Eigen::VectorXd>& u, std::vector<Eigen::MatrixXd>& cov,
    std::vector<double>& pi, const std::vector<int>& S = { 0, 1, 2 }, int n_iter = 100)
{
    Eigen::MatrixXd O_array(O.size(), O[0].size());
    for (size_t i = 0; i < O.size(); ++i) {
        for (size_t j = 0; j < O[i].size(); ++j) {
            O_array(i, j) = O[i][j];
        }
    }
    int M = trans.rows();
    int T = O_array.rows();
    double likelihood_hist = -std::numeric_limits<double>::infinity();
    Eigen::VectorXi best_path(T);

    for (int n = 0; n < n_iter; ++n) {
        // Estimation step
        auto result = scaled_forward(O, trans, u, cov, pi, S);
        Eigen::MatrixXd alpha = result.first;
        Eigen::VectorXd G = result.second;
        Eigen::MatrixXd beta = scaled_backward(O, trans, u, cov, pi, S);

        Eigen::MatrixXd b(trans.rows(), O_array.rows());
        for (int i = 0; i < trans.rows(); ++i) {
            for (int j = 0; j < O_array.rows(); ++j) {
                b(i, j) = emission(i, O[j], u, cov, S);
            }
        }
        //std::cout << "b" << b << std::endl;
        Eigen::Tensor<double, 3> xi(M, M, T - 1);
        for (int t = 0; t < T - 1; ++t) {
            //std::cout << "beta_t+1" << beta.row(t + 1).transpose() << std::endl;
            //std::cout << b.col(t + 1) << std::endl;
            double denominator = alpha.row(t) * trans * (b.col(t + 1).cwiseProduct(beta.row(t+1).transpose()));
            //std::cout << denominator << std::endl;
            for (int i = 0; i < M; ++i) {
                Eigen::VectorXd numerator = alpha(t,i) * (trans.row(i).transpose().cwiseProduct(b.col(t + 1))).cwiseProduct(beta.row(t + 1).transpose());
                for (int j = 0; j < M; ++j) {
                    xi.coeffRef(i, j, t) = numerator(j) / denominator;
                }
            }
        }
        //std::cout << "xi" << xi << std::endl;
        Eigen::Tensor<double, 2> gamma_temp = xi.sum(Eigen::array<int, 1>({ 1 }));
        Eigen::MatrixXd gamma(M,T-1);
        for (int i = 0; i < M; ++i) {
            for (int t = 0; t < T - 1; ++t) {
                gamma(i, t) = gamma_temp(i, t);
            }
        }
        //std::cout << "gamma" << gamma << std::endl;
        // Maximization step
        Eigen::Tensor<double, 2> trans_temp1 = xi.sum(Eigen::array<int, 1>({ 2 }));
        for (int i = 0; i < M; ++i) {
            double trans_temp2 = gamma.row(i).sum();
            for (int j = 0; j < M; ++j) {
                trans(i, j) = trans_temp1(i,j)/trans_temp2;
            }
        }
        //std::cout << "trans" << trans << std::endl;
        /*
        Eigen::Tensor<double, 2> trans_temp = xi.sum(Eigen::array<int, 1>({ 2 })) / gamma.rowwise().sum().transpose().array().rowwise().replicate(M);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                trans(i, j) = trans_temp(i, j);
            }
        }
        */
        gamma.conservativeResize(Eigen::NoChange, gamma.cols() + 1);
        Eigen::Tensor<double, 1> gamma_temp1 = xi.chip(T - 2, 2).sum(Eigen::array<int, 1>({ 0 }));
        for (int i = 0; i < M; ++i) {
            gamma(i,T - 1) = gamma_temp1(i);
        }
        //std::cout << "gamma" << gamma << std::endl;
        for (int t = 0; t < T; ++t) {
            double best_path_temp = gamma.col(t).maxCoeff();
            for (int i = 0; i < M; ++i) {
                //std::cout << best_path_temp << std::endl;
                //std::cout << gamma(i, t) << std::endl;
                if (gamma(i, t) == best_path_temp) {
                    best_path(t) = i;
                }
            }
        }
        //std::cout << "best path" << best_path << std::endl;
        //std::cout << "likelihood_hist" << likelihood_hist << std::endl;
        double log_likelihood = (G.array().log()).sum();
        if (log_likelihood < likelihood_hist) {
            std::cout << "log_likelihood decreased" << std::endl;
            break;
        }
        if (log_likelihood - likelihood_hist < 0.000001) {
            std::cout << n << " steps' log likelihood: " << log_likelihood << " converged" << std::endl;
            break;
        }
        likelihood_hist = log_likelihood;
        std::cout << "likelihood_hist" << likelihood_hist << std::endl;
        Eigen::VectorXd pi_temp = gamma.col(0).array();
        //std::cout << "pi" << pi_temp << std::endl;
        //std::cout << "pi" << pi << std::endl;
        for (int i = 0; i < M; ++i) {
            pi[i] = pi_temp(i);
        }
        //std::cout << "pi" << pi[0] << std::endl;
        for (int i = 0; i < M; ++i) {
            u[i] = (gamma.row(i) * O_array) / gamma.row(i).sum();
        }
        //std::cout << "u" << u[0] << std::endl;
        //std::cout << "u" << u[1] << std::endl;
        //std::cout << "u" << u[2] << std::endl;
        for (int i = 0; i < M; ++i) {
            //std::cout << O_array.cols() << std::endl;
            Eigen::MatrixXd sigma(O_array.cols(), O_array.cols());
            sigma.setZero();
            for (int j = 0; j < T; ++j) {
                //std::cout << O_array.row(j).transpose() - u[i] << std::endl;
                sigma += (O_array.row(j).transpose() - u[i])*(O_array.row(j).transpose() - u[i]).transpose() * gamma(i, j);
            }
            cov[i] = sigma *(1/ gamma.row(i).sum());
        }
        /*
        std::cout << "cov1" << cov[0] << std::endl;
        std::cout << "cov2" << cov[1] << std::endl;
        std::cout << "cov3" << cov[2] << std::endl;
        */
    }
    /*
    Eigen::MatrixXd trans_result(trans.rows(), trans.cols());
    for (int i = 0; i < trans.rows(); ++i) {
        for (int j = 0; j < trans.cols(); ++j) {
            trans_result(i, j) = trans(i, j);
        }
    }
    */
    //std::vector<int> best_path_result(best_path.data(), best_path.data() + best_path.size());

    return std::make_tuple(trans, u, cov, pi, best_path);
}

