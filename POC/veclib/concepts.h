#pragma once


namespace zeno::ztd {
inline namespace concepts {

template <class T1>
concept has_bit_not = requires(T1 t1) {
    ~t1;
};

template <class T1>
concept has_unary_plus = requires(T1 t1) {
    +t1;
};

template <class T1>
concept has_negate = requires(T1 t1) {
    -t1;
};

template <class T1, class T2>
concept has_bit_or = requires(T1 t1, T2 t2) {
    t1 | t2;
};

template <class T1, class T2>
concept has_bit_and = requires(T1 t1, T2 t2) {
    t1 & t2;
};

template <class T1, class T2>
concept has_bit_xor = requires(T1 t1, T2 t2) {
    t1 ^ t2;
};

template <class T1, class T2>
concept has_bit_shl = requires(T1 t1, T2 t2) {
    t1 << t2;
};

template <class T1, class T2>
concept has_bit_shr = requires(T1 t1, T2 t2) {
    t1 >> t2;
};

template <class T1, class T2>
concept has_plus = requires(T1 t1, T2 t2) {
    t1 + t2;
};

template <class T1, class T2>
concept has_minus = requires(T1 t1, T2 t2) {
    t1 - t2;
};

template <class T1, class T2>
concept has_times = requires(T1 t1, T2 t2) {
    t1 * t2;
};

template <class T1, class T2>
concept has_divide = requires(T1 t1, T2 t2) {
    t1 / t2;
};

template <class T1, class T2>
concept has_modolus = requires(T1 t1, T2 t2) {
    t1 % t2;
};

template <class T1, class T2>
concept has_assign = requires(T1 t1, T2 t2) {
    t1 |= t2;
};

template <class T1, class T2>
concept has_bit_or_assign = requires(T1 t1, T2 t2) {
    t1 |= t2;
};

template <class T1, class T2>
concept has_bit_and_assign = requires(T1 t1, T2 t2) {
    t1 &= t2;
};

template <class T1, class T2>
concept has_bit_xor_assign = requires(T1 t1, T2 t2) {
    t1 ^= t2;
};

template <class T1, class T2>
concept has_bit_shl_assign = requires(T1 t1, T2 t2) {
    t1 <<= t2;
};

template <class T1, class T2>
concept has_bit_shr_assign = requires(T1 t1, T2 t2) {
    t1 >>= t2;
};

template <class T1, class T2>
concept has_plus_assign = requires(T1 t1, T2 t2) {
    t1 += t2;
};

template <class T1, class T2>
concept has_minus_assign = requires(T1 t1, T2 t2) {
    t1 -= t2;
};

template <class T1, class T2>
concept has_times_assign = requires(T1 t1, T2 t2) {
    t1 *= t2;
};

template <class T1, class T2>
concept has_divide_assign = requires(T1 t1, T2 t2) {
    t1 /= t2;
};

template <class T1, class T2>
concept has_modolus_assign = requires(T1 t1, T2 t2) {
    t1 %= t2;
};

template <class T1, class T2>
concept has_equal = requires(T1 t1, T2 t2) {
    t1 == t2;
};

template <class T1, class T2>
concept has_not_equal = requires(T1 t1, T2 t2) {
    t1 != t2;
};

template <class T1, class T2>
concept has_less = requires(T1 t1, T2 t2) {
    t1 < t2;
};

template <class T1, class T2>
concept has_greater = requires(T1 t1, T2 t2) {
    t1 > t2;
};

template <class T1, class T2>
concept has_less_equal = requires(T1 t1, T2 t2) {
    t1 <= t2;
};

template <class T1, class T2>
concept has_greater_equal = requires(T1 t1, T2 t2) {
    t1 >= t2;
};

template <class T1, class T2>
concept has_three_way = requires(T1 t1, T2 t2) {
    t1 <=> t2;
};

template <class T1, class T2>
concept has_logical_not = requires(T1 t1) {
    !t1;
};

}
}
