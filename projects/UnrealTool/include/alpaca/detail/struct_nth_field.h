#pragma once
#include <alpaca/detail/aggregate_arity.h>
#include <cstdint>

namespace alpaca {

namespace detail {

template <std::size_t index, typename type,
          std::size_t arity = aggregate_arity<std::remove_cv_t<type>>::size()>
constexpr decltype(auto) get(type &value) noexcept {

  if constexpr (arity == 1) {
    auto &[p1] = value;
    if constexpr (index == 0) {
      return (p1);
    } else {
      return;
    }
  } else if constexpr (arity == 2) {
    auto &[p1, p2] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else {
      return;
    }
  } else if constexpr (arity == 3) {
    auto &[p1, p2, p3] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else {
      return;
    }
  } else if constexpr (arity == 4) {
    auto &[p1, p2, p3, p4] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else {
      return;
    }
  } else if constexpr (arity == 5) {
    auto &[p1, p2, p3, p4, p5] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else {
      return;
    }
  } else if constexpr (arity == 6) {
    auto &[p1, p2, p3, p4, p5, p6] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else {
      return;
    }
  } else if constexpr (arity == 7) {
    auto &[p1, p2, p3, p4, p5, p6, p7] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else {
      return;
    }
  } else if constexpr (arity == 8) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else {
      return;
    }
  } else if constexpr (arity == 9) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else {
      return;
    }
  } else if constexpr (arity == 10) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else {
      return;
    }
  } else if constexpr (arity == 11) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else {
      return;
    }
  } else if constexpr (arity == 12) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else {
      return;
    }
  } else if constexpr (arity == 13) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else {
      return;
    }
  } else if constexpr (arity == 14) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else {
      return;
    }
  } else if constexpr (arity == 15) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15] =
        value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else {
      return;
    }
  } else if constexpr (arity == 16) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else {
      return;
    }
  } else if constexpr (arity == 17) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else {
      return;
    }
  } else if constexpr (arity == 18) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else {
      return;
    }
  } else if constexpr (arity == 19) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else {
      return;
    }
  } else if constexpr (arity == 20) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else {
      return;
    }
  } else if constexpr (arity == 21) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else {
      return;
    }
  } else if constexpr (arity == 22) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else {
      return;
    }
  } else if constexpr (arity == 23) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else {
      return;
    }
  } else if constexpr (arity == 24) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else {
      return;
    }
  } else if constexpr (arity == 25) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else {
      return;
    }
  } else if constexpr (arity == 26) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else {
      return;
    }
  } else if constexpr (arity == 27) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else {
      return;
    }
  } else if constexpr (arity == 28) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28] =
        value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else {
      return;
    }
  } else if constexpr (arity == 29) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28,
           p29] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else {
      return;
    }
  } else if constexpr (arity == 30) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else {
      return;
    }
  } else if constexpr (arity == 31) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else {
      return;
    }
  } else if constexpr (arity == 32) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else {
      return;
    }
  } else if constexpr (arity == 33) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else {
      return;
    }
  } else if constexpr (arity == 34) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else {
      return;
    }
  } else if constexpr (arity == 35) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else {
      return;
    }
  } else if constexpr (arity == 36) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else {
      return;
    }
  } else if constexpr (arity == 37) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else {
      return;
    }
  } else if constexpr (arity == 38) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else {
      return;
    }
  } else if constexpr (arity == 39) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else {
      return;
    }
  } else if constexpr (arity == 40) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else {
      return;
    }
  } else if constexpr (arity == 41) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else {
      return;
    }
  } else if constexpr (arity == 42) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42] =
        value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else {
      return;
    }
  } else if constexpr (arity == 43) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42,
           p43] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else {
      return;
    }
  } else if constexpr (arity == 44) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else {
      return;
    }
  } else if constexpr (arity == 45) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else {
      return;
    }
  } else if constexpr (arity == 46) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else {
      return;
    }
  } else if constexpr (arity == 47) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else {
      return;
    }
  } else if constexpr (arity == 48) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else {
      return;
    }
  } else if constexpr (arity == 49) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else {
      return;
    }
  } else if constexpr (arity == 50) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else {
      return;
    }
  } else if constexpr (arity == 51) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else {
      return;
    }
  } else if constexpr (arity == 52) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else {
      return;
    }
  } else if constexpr (arity == 53) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else {
      return;
    }
  } else if constexpr (arity == 54) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else {
      return;
    }
  } else if constexpr (arity == 55) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else {
      return;
    }
  } else if constexpr (arity == 56) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56] =
        value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else {
      return;
    }
  } else if constexpr (arity == 57) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56,
           p57] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else {
      return;
    }
  } else if constexpr (arity == 58) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else {
      return;
    }
  } else if constexpr (arity == 59) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else {
      return;
    }
  } else if constexpr (arity == 60) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else {
      return;
    }
  } else if constexpr (arity == 61) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else {
      return;
    }
  } else if constexpr (arity == 62) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else {
      return;
    }
  } else if constexpr (arity == 63) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else {
      return;
    }
  } else if constexpr (arity == 64) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else {
      return;
    }
  } else if constexpr (arity == 65) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else {
      return;
    }
  } else if constexpr (arity == 66) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else {
      return;
    }
  } else if constexpr (arity == 67) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else {
      return;
    }
  } else if constexpr (arity == 68) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else {
      return;
    }
  } else if constexpr (arity == 69) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else {
      return;
    }
  } else if constexpr (arity == 70) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70] =
        value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else {
      return;
    }
  } else if constexpr (arity == 71) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70,
           p71] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else {
      return;
    }
  } else if constexpr (arity == 72) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else {
      return;
    }
  } else if constexpr (arity == 73) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else {
      return;
    }
  } else if constexpr (arity == 74) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else {
      return;
    }
  } else if constexpr (arity == 75) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else {
      return;
    }
  } else if constexpr (arity == 76) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else {
      return;
    }
  } else if constexpr (arity == 77) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else {
      return;
    }
  } else if constexpr (arity == 78) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else {
      return;
    }
  } else if constexpr (arity == 79) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else {
      return;
    }
  } else if constexpr (arity == 80) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else {
      return;
    }
  } else if constexpr (arity == 81) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else {
      return;
    }
  } else if constexpr (arity == 82) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else {
      return;
    }
  } else if constexpr (arity == 83) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else {
      return;
    }
  } else if constexpr (arity == 84) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84] =
        value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else {
      return;
    }
  } else if constexpr (arity == 85) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84,
           p85] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else {
      return;
    }
  } else if constexpr (arity == 86) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else {
      return;
    }
  } else if constexpr (arity == 87) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else {
      return;
    }
  } else if constexpr (arity == 88) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else {
      return;
    }
  } else if constexpr (arity == 89) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else {
      return;
    }
  } else if constexpr (arity == 90) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else {
      return;
    }
  } else if constexpr (arity == 91) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90, p91] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else if constexpr (index == 90) {
      return (p91);
    } else {
      return;
    }
  } else if constexpr (arity == 92) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90, p91, p92] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else if constexpr (index == 90) {
      return (p91);
    } else if constexpr (index == 91) {
      return (p92);
    } else {
      return;
    }
  } else if constexpr (arity == 93) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90, p91, p92, p93] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else if constexpr (index == 90) {
      return (p91);
    } else if constexpr (index == 91) {
      return (p92);
    } else if constexpr (index == 92) {
      return (p93);
    } else {
      return;
    }
  } else if constexpr (arity == 94) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90, p91, p92, p93, p94] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else if constexpr (index == 90) {
      return (p91);
    } else if constexpr (index == 91) {
      return (p92);
    } else if constexpr (index == 92) {
      return (p93);
    } else if constexpr (index == 93) {
      return (p94);
    } else {
      return;
    }
  } else if constexpr (arity == 95) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90, p91, p92, p93, p94, p95] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else if constexpr (index == 90) {
      return (p91);
    } else if constexpr (index == 91) {
      return (p92);
    } else if constexpr (index == 92) {
      return (p93);
    } else if constexpr (index == 93) {
      return (p94);
    } else if constexpr (index == 94) {
      return (p95);
    } else {
      return;
    }
  } else if constexpr (arity == 96) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90, p91, p92, p93, p94, p95, p96] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else if constexpr (index == 90) {
      return (p91);
    } else if constexpr (index == 91) {
      return (p92);
    } else if constexpr (index == 92) {
      return (p93);
    } else if constexpr (index == 93) {
      return (p94);
    } else if constexpr (index == 94) {
      return (p95);
    } else if constexpr (index == 95) {
      return (p96);
    } else {
      return;
    }
  } else if constexpr (arity == 97) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90, p91, p92, p93, p94, p95, p96, p97] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else if constexpr (index == 90) {
      return (p91);
    } else if constexpr (index == 91) {
      return (p92);
    } else if constexpr (index == 92) {
      return (p93);
    } else if constexpr (index == 93) {
      return (p94);
    } else if constexpr (index == 94) {
      return (p95);
    } else if constexpr (index == 95) {
      return (p96);
    } else if constexpr (index == 96) {
      return (p97);
    } else {
      return;
    }
  } else if constexpr (arity == 98) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90, p91, p92, p93, p94, p95, p96, p97, p98] =
        value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else if constexpr (index == 90) {
      return (p91);
    } else if constexpr (index == 91) {
      return (p92);
    } else if constexpr (index == 92) {
      return (p93);
    } else if constexpr (index == 93) {
      return (p94);
    } else if constexpr (index == 94) {
      return (p95);
    } else if constexpr (index == 95) {
      return (p96);
    } else if constexpr (index == 96) {
      return (p97);
    } else if constexpr (index == 97) {
      return (p98);
    } else {
      return;
    }
  } else if constexpr (arity == 99) {
    auto &[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
           p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
           p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
           p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57,
           p58, p59, p60, p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71,
           p72, p73, p74, p75, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85,
           p86, p87, p88, p89, p90, p91, p92, p93, p94, p95, p96, p97, p98,
           p99] = value;
    if constexpr (index == 0) {
      return (p1);
    } else if constexpr (index == 1) {
      return (p2);
    } else if constexpr (index == 2) {
      return (p3);
    } else if constexpr (index == 3) {
      return (p4);
    } else if constexpr (index == 4) {
      return (p5);
    } else if constexpr (index == 5) {
      return (p6);
    } else if constexpr (index == 6) {
      return (p7);
    } else if constexpr (index == 7) {
      return (p8);
    } else if constexpr (index == 8) {
      return (p9);
    } else if constexpr (index == 9) {
      return (p10);
    } else if constexpr (index == 10) {
      return (p11);
    } else if constexpr (index == 11) {
      return (p12);
    } else if constexpr (index == 12) {
      return (p13);
    } else if constexpr (index == 13) {
      return (p14);
    } else if constexpr (index == 14) {
      return (p15);
    } else if constexpr (index == 15) {
      return (p16);
    } else if constexpr (index == 16) {
      return (p17);
    } else if constexpr (index == 17) {
      return (p18);
    } else if constexpr (index == 18) {
      return (p19);
    } else if constexpr (index == 19) {
      return (p20);
    } else if constexpr (index == 20) {
      return (p21);
    } else if constexpr (index == 21) {
      return (p22);
    } else if constexpr (index == 22) {
      return (p23);
    } else if constexpr (index == 23) {
      return (p24);
    } else if constexpr (index == 24) {
      return (p25);
    } else if constexpr (index == 25) {
      return (p26);
    } else if constexpr (index == 26) {
      return (p27);
    } else if constexpr (index == 27) {
      return (p28);
    } else if constexpr (index == 28) {
      return (p29);
    } else if constexpr (index == 29) {
      return (p30);
    } else if constexpr (index == 30) {
      return (p31);
    } else if constexpr (index == 31) {
      return (p32);
    } else if constexpr (index == 32) {
      return (p33);
    } else if constexpr (index == 33) {
      return (p34);
    } else if constexpr (index == 34) {
      return (p35);
    } else if constexpr (index == 35) {
      return (p36);
    } else if constexpr (index == 36) {
      return (p37);
    } else if constexpr (index == 37) {
      return (p38);
    } else if constexpr (index == 38) {
      return (p39);
    } else if constexpr (index == 39) {
      return (p40);
    } else if constexpr (index == 40) {
      return (p41);
    } else if constexpr (index == 41) {
      return (p42);
    } else if constexpr (index == 42) {
      return (p43);
    } else if constexpr (index == 43) {
      return (p44);
    } else if constexpr (index == 44) {
      return (p45);
    } else if constexpr (index == 45) {
      return (p46);
    } else if constexpr (index == 46) {
      return (p47);
    } else if constexpr (index == 47) {
      return (p48);
    } else if constexpr (index == 48) {
      return (p49);
    } else if constexpr (index == 49) {
      return (p50);
    } else if constexpr (index == 50) {
      return (p51);
    } else if constexpr (index == 51) {
      return (p52);
    } else if constexpr (index == 52) {
      return (p53);
    } else if constexpr (index == 53) {
      return (p54);
    } else if constexpr (index == 54) {
      return (p55);
    } else if constexpr (index == 55) {
      return (p56);
    } else if constexpr (index == 56) {
      return (p57);
    } else if constexpr (index == 57) {
      return (p58);
    } else if constexpr (index == 58) {
      return (p59);
    } else if constexpr (index == 59) {
      return (p60);
    } else if constexpr (index == 60) {
      return (p61);
    } else if constexpr (index == 61) {
      return (p62);
    } else if constexpr (index == 62) {
      return (p63);
    } else if constexpr (index == 63) {
      return (p64);
    } else if constexpr (index == 64) {
      return (p65);
    } else if constexpr (index == 65) {
      return (p66);
    } else if constexpr (index == 66) {
      return (p67);
    } else if constexpr (index == 67) {
      return (p68);
    } else if constexpr (index == 68) {
      return (p69);
    } else if constexpr (index == 69) {
      return (p70);
    } else if constexpr (index == 70) {
      return (p71);
    } else if constexpr (index == 71) {
      return (p72);
    } else if constexpr (index == 72) {
      return (p73);
    } else if constexpr (index == 73) {
      return (p74);
    } else if constexpr (index == 74) {
      return (p75);
    } else if constexpr (index == 75) {
      return (p76);
    } else if constexpr (index == 76) {
      return (p77);
    } else if constexpr (index == 77) {
      return (p78);
    } else if constexpr (index == 78) {
      return (p79);
    } else if constexpr (index == 79) {
      return (p80);
    } else if constexpr (index == 80) {
      return (p81);
    } else if constexpr (index == 81) {
      return (p82);
    } else if constexpr (index == 82) {
      return (p83);
    } else if constexpr (index == 83) {
      return (p84);
    } else if constexpr (index == 84) {
      return (p85);
    } else if constexpr (index == 85) {
      return (p86);
    } else if constexpr (index == 86) {
      return (p87);
    } else if constexpr (index == 87) {
      return (p88);
    } else if constexpr (index == 88) {
      return (p89);
    } else if constexpr (index == 89) {
      return (p90);
    } else if constexpr (index == 90) {
      return (p91);
    } else if constexpr (index == 91) {
      return (p92);
    } else if constexpr (index == 92) {
      return (p93);
    } else if constexpr (index == 93) {
      return (p94);
    } else if constexpr (index == 94) {
      return (p95);
    } else if constexpr (index == 95) {
      return (p96);
    } else if constexpr (index == 96) {
      return (p97);
    } else if constexpr (index == 97) {
      return (p98);
    } else if constexpr (index == 98) {
      return (p99);
    } else {
      return;
    }
  } else /* extend it by yourself for higher arities */ {
    return;
  }
}
} // namespace detail

} // namespace alpaca