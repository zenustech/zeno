#pragma once

namespace scheme {

template <typename T> constexpr T HJ_WENO3(T fm, T fi, T fp, T fpp, T u, T dh) {
    const int sgn = (u > 0) ? 1 : -1;
    const T D = -sgn * dh;
    const T eps = (T)1e-10;
    const T df3 = -(-fm + (T)3.0 * fi - (T)3.0 * fp + fpp) / ((T)2.0 * D);

    const T a = (fi - (T)2.0 * fp + fpp);
    const T b = (fp - (T)2.0 * fi + fm);

    const T r = (a * a + eps) / (b * b + eps);

    const T dfc = (fp - fm) / ((T)2.0 * D);

    return dfc + df3 / ((T)1.0 + (T)2.0 * r * r);
}

template <typename T> constexpr T central_diff_2nd(T fm, T f, T fp, T dh) {
    return (fp - (T)2.0 * f + fm) / (dh * dh);
}

template <typename T> constexpr T minmod(const T a, const T b) {
    const T sgn = a >= 0. ? 1.0 : -1.0;
    return sgn * fmax(0.0, fmin(sgn * a, sgn * b));
}

template <typename T> constexpr T TVD_MUSCL3(const T fdw, const T fup, const T fup2) {
    const T b = (T)4.0;
    const T dfp = fup2 - fup;
    const T dfm = fup - fdw;
    return fup - (T)0.25 * ((T)4.0 / (T)3.0 * minmod(dfm, dfp * b) + (T)2.0 / (T)3.0 * minmod(dfp, dfm * b));
}

template <typename T> constexpr T face_fraction(T ls_bl, T ls_br, T ls_tl, T ls_tr) {

    return (T)0;
}

} // namespace scheme