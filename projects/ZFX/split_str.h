#pragma once

#include <string>
#include <vector>
#include <sstream>

static std::vector<std::string> split_str(std::string const &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}
