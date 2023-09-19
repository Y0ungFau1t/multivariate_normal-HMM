#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::vector<double>> readCSV1(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;

    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            // 将每个单元格转换为 double 并添加到当前行中
            row.push_back(std::stod(cell));
        }

        // 将当前行添加到数据集中
        data.push_back(row);
    }

    file.close();
    return data;
}

Eigen::VectorXi readCSV2(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<int> data;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            data.push_back(std::stoi(cell));
        }
    }

    Eigen::VectorXi vectorX(data.size());

    for (int i = 0; i < data.size(); ++i) {
        vectorX(i) = data[i];
    }

    return vectorX;
}