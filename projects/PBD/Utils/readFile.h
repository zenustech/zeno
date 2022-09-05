#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

/**
 * @brief 读取txt文件到场中（如pos lines等）
 * 
 * @tparam T 场类型，如std::vector<vec3f>
 * @param absPath txt文件的绝对路径，注意windows路径用\\
 * @param saveTo 要保存到的场如pos, lines, tets等
 */
template<typename T>
void readFile(const std::string &absPath, T &saveTo)
{
    std::ifstream in(absPath);
    std::string line;

    using TT = typename T::value_type;
    TT temp;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        for (int i = 0; i < temp.size(); i++)
            ss >> temp[i];
        saveTo.emplace_back(temp);
    }
    in.close();
}