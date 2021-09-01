#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <array>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <type_traits>
#include <unordered_set>
#include <string>
#include <cassert>
#include <iostream>

// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp));

namespace ZenEulerGas {
template <typename T, int dim>
using Vector = Eigen::Matrix<T, dim, 1, 0, dim, 1>;

template <typename T, int n, int m>
using Array = Eigen::Array<T, n, m, 0, n, m>;

template <typename T, int n, int m>
using Matrix = Eigen::Matrix<T, n, m, 0, n, m>;

template <typename DerivedV>
using Field = std::vector<DerivedV, Eigen::aligned_allocator<DerivedV>>;

} // namespace ZenEulerGas

#endif