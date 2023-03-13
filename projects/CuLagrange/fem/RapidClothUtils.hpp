#pragma once 
#include "zensim/math/Vec.h"

namespace zs {
  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 2,
                                         VecT::template range_t<1>::value == 2,
                                         std::is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto adjoint(const VecInterface<VecT> &A) noexcept {
    using T = typename VecT::value_type;
    auto ret = VecT::zeros();
    ret(0, 0) = A(1, 1);
    ret(1, 0) = -A(1, 0);
    ret(0, 1) = -A(0, 1);
    ret(1, 1) = A(0, 0);
    return ret;
  }
  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 3,
                                         VecT::template range_t<1>::value == 3,
                                         std::is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto adjoint(const VecInterface<VecT> &A) noexcept {
    using T = typename VecT::value_type;
    auto ret = VecT::zeros();
    ret(0, 0) = detail::cofactor<0, 0>(A);
    ret(0, 1) = detail::cofactor<1, 0>(A);
    ret(0, 2) = detail::cofactor<2, 0>(A);
    
    T c01 = detail::cofactor<0, 1>(A);
    T c11 = detail::cofactor<1, 1>(A);
    T c02 = detail::cofactor<0, 2>(A);
    ret(1, 2) = detail::cofactor<2, 1>(A);
    ret(2, 1) = detail::cofactor<1, 2>(A);
    ret(2, 2) = detail::cofactor<2, 2>(A);
    ret(1, 0) = c01;
    ret(1, 1) = c11;
    ret(2, 0) = c02;
    return ret;
  }
  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 4,
                                         VecT::template range_t<1>::value == 4,
                                         std::is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto adjoint(const VecInterface<VecT> &A) noexcept {
    // using T = typename VecT::value_type;
    auto ret = VecT::zeros();
    ret(0, 0) = detail::cofactor<0, 0>(A);
    ret(1, 0) = -detail::cofactor<0, 1>(A);
    ret(2, 0) = detail::cofactor<0, 2>(A);
    ret(3, 0) = -detail::cofactor<0, 3>(A);

    ret(0, 2) = detail::cofactor<2, 0>(A);
    ret(1, 2) = -detail::cofactor<2, 1>(A);
    ret(2, 2) = detail::cofactor<2, 2>(A);
    ret(3, 2) = -detail::cofactor<2, 3>(A);

    ret(0, 1) = -detail::cofactor<1, 0>(A);
    ret(1, 1) = detail::cofactor<1, 1>(A);
    ret(2, 1) = -detail::cofactor<1, 2>(A);
    ret(3, 1) = detail::cofactor<1, 3>(A);

    ret(0, 3) = -detail::cofactor<3, 0>(A);
    ret(1, 3) = detail::cofactor<3, 1>(A);
    ret(2, 3) = -detail::cofactor<3, 2>(A);
    ret(3, 3) = detail::cofactor<3, 3>(A);
    return ret; 
  }
}