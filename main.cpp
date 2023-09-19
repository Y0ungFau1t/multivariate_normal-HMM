//#include "scale_forward.h"
//#include "scale_backward.h"
#include "scaled_baum_welch.h"
#include "read.h"

int main(){
    std::string filename1 = "observation_simulation.csv";
    std::vector<std::vector<double>> O = readCSV1(filename1);
    std::string filename2 = "hiddenstates_simulation.csv";
    Eigen::VectorXi real_path = readCSV2(filename2);
    std::cout << "长度" << real_path.rows()<<std::endl;
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(3, 3);
    T << 0.8, 0.1, 0.1,
        0.1, 0.8, 0.1,
        0.1, 0.1, 0.8;
    Eigen::VectorXd u1(4);
    u1 << 0.86, 0.988, -1.006, 0.025;
    Eigen::VectorXd u2(4);
    u2 << -0.897, 1.054, 1.00871, 0;
    Eigen::VectorXd u3(4);
    u3 << 1.154, -0.98659, 1.0001, 0.00687;
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
    std::vector<MatrixXd> cov = { cov1,cov1,cov1 };
    std::vector<double> pi = {1.0/3,1.0 / 3,1.0 / 3 };
    std::cout << "pi" << pi[0] << std::endl;
    std::cout << "u" << u[0] << std::endl;
    std::vector<int> S = { 0,1,2 };
    std::cout << "S" << S[1] << std::endl;
    /*
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> result = scaled_forward(O, T, u, cov, pi, S);
    Eigen::MatrixXd alpha_hat = result.first;
    Eigen::VectorXd G = result.second;
    // Print the alpha_hat matrix
    std::cout << "Alpha Hat Matrix:" << std::endl;
    std::cout << alpha_hat << std::endl;
    std::cout << "G " << G << std::endl;
    Eigen::MatrixXd beta_hat = scaled_backward(O, T, u, cov, pi, S);
    std::cout << "Beta Hat Matrix:" << std::endl;
    std::cout << beta_hat << std::endl;
    */
    auto result1 = scaled_baum_welch(O, T, u, cov, pi, S, 1000);
    auto best_path = std::get<4>(result1);
    if (real_path.rows() == best_path.rows()) {
        std::cout << "长度相同" << std::endl;
    }
    double acc = 0;
    for (int i = 0; i < real_path.rows(); ++i) {
        if (real_path(i) == best_path(i)) {
            ++acc;
        }
    }
    std::cout << "acc"<< acc/real_path.rows();
    return 0;
}