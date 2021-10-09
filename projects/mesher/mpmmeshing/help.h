#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include <fstream>
#include <sstream>
#include <string.h>

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <ctime>

std::vector<int> ArrangeCore(std::vector<int> a);

#include <tuple>

typedef std::tuple<int, int, int> key_f;

struct key_hash : public std::function<std::size_t(key_f)>
{
    std::size_t operator()(const key_f &k) const
    {
        return std::get<0>(k) ^ std::get<1>(k) ^ std::get<2>(k);
    }
};

struct key_equal : public std::function<bool(key_f, key_f)>
{
    bool operator()(const key_f &v0, const key_f &v1) const
    {
        return (std::get<0>(v0) == std::get<0>(v1) && std::get<1>(v0) == std::get<1>(v1) && std::get<2>(v0) == std::get<2>(v1));
    }
};

struct hash_pair
{
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2> &p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};

Eigen::Vector3i sort(int x, int y, int z);
Eigen::Vector2i sort(int x, int y);