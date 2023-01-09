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
    return sgn * zs::max((T)0, zs::min(sgn * a, sgn * b));
}

template <typename T> constexpr T TVD_MUSCL3(const T fdw, const T fup, const T fup2) {
    const T b = (T)4.0;
    const T dfp = fup2 - fup;
    const T dfm = fup - fdw;
    return fup - (T)0.25 * ((T)4.0 / (T)3.0 * minmod(dfm, dfp * b) + (T)2.0 / (T)3.0 * minmod(dfp, dfm * b));
}

template <typename T> constexpr T clip(const T min, const T f, const T max) {
    return (f > max) ? max : (f < min) ? min : f;
}

template <typename T> constexpr T line_fraction(T ls_0, T ls_1) {
    T p{};

    if (ls_0 * ls_1 < 0.) {
        p = ls_0 / (ls_0 - ls_1);
        if (ls_0 < 0.) {
            p = 1.f - p;
        }
    } else {
        p = (ls_0 > 0. || ls_1 > 0.);
    }

    return p;
}

template <typename T> constexpr T line_area(T nx, T ny, T alpha) {
    T a{}, v{}, area{};

    alpha += (nx + ny) / (T)2.0;
    if (nx < 0.) {
        alpha -= nx;
        nx = -nx;
    }
    if (ny < 0.) {
        alpha -= ny;
        ny = -ny;
    }

    if (alpha <= 0.)
        return 0.;

    if (alpha >= nx + ny)
        return 1.;

    if (nx < 1e-10)
        area = alpha / ny;
    else if (ny < 1e-10)
        area = alpha / nx;
    else {
        v = alpha * alpha;

        a = alpha - nx;
        if (a > 0.)
            v -= a * a;

        a = alpha - ny;
        if (a > 0.)
            v -= a * a;

        area = v / (2. * nx * ny);
    }

    return clip((T)0., area, (T)1.0);
}

template <typename T> constexpr T face_fraction(T ls_bl, T ls_br, T ls_tl, T ls_tr) {
    T px[2] = {}, py[2] = {};
    px[0] = line_fraction(ls_bl, ls_tl);
    px[1] = line_fraction(ls_br, ls_tr);
    py[0] = line_fraction(ls_bl, ls_br);
    py[1] = line_fraction(ls_tl, ls_tr);

    T ls_x[2] = {ls_bl, ls_br};
    T ls_y[2] = {ls_bl, ls_tl};

    T n[2] = {};
    n[0] = px[0] - px[1];
    n[1] = py[0] - py[1];

    T nn = zs::abs(n[0]) + zs::abs(n[1]);
    T s_z{};

    if (nn < 1e-10) {
        s_z = px[0];
    } else {
        n[0] /= nn;
        n[1] /= nn;

        T alpha = 0.;
        int ni = 0;

        for (int i = 0; i <= 1; i++) {
            if (px[i] > 0. && px[i] < 1.) {
                int sign = ls_x[i] > 0. ? 1 : -1;
                T a = sign * (px[i] - (T)0.5);
                alpha += n[0] * (i - (T)0.5) + n[1] * a;
                ni++;
            }

            if (py[i] > 0. && py[i] < 1.) {
                int sign = ls_y[i] > 0. ? 1 : -1;
                T a = sign * (py[i] - (T)0.5);
                alpha += n[0] * a + n[1] * (i - (T)0.5);
                ni++;
            }
        }

        if (ni == 0) {
            s_z = zs::max(px[0], py[0]);
        } else if (ni != 4) {
            s_z = line_area(n[0], n[1], alpha / ni);
        } else {
            s_z = 0.;
        }
    }

    return s_z;
}

} // namespace scheme