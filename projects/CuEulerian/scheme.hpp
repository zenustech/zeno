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

} // namespace scheme