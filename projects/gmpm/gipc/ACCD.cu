#include "ACCD.cuh"
#include "gpu_eigen_libs.cuh"
#include <cmath>
#include <stdio.h>
template <class F>
__device__ __host__
inline F __m_max(F a, F b) {
    return a > b ? a : b;
}

template <class F>
__device__ __host__
inline F __m_min(F a, F b) {
    return a > b ? b : a;
}

__device__
int _dType_point_triangle(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 basis0 = __GEIGEN__::__minus(v2, v1);
    double3 basis1 = __GEIGEN__::__minus(v3, v1);
    double3 basis2 = __GEIGEN__::__minus(v0, v1);

    const double3 nVec = __GEIGEN__::__v_vec_cross(basis0, basis1);

    basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
    __GEIGEN__::Matrix3x3d D, D1, D2;

    __GEIGEN__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
    __GEIGEN__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
    __GEIGEN__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

    double2 param[3];
    param[0].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
    param[0].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

    if (param[0].x > 0 && param[0].x < 1 && param[0].y >= 0) {
        return 3; // PE v1v2
    }
    else {
        basis0 = __GEIGEN__::__minus(v3, v2);
        basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
        basis2 = __GEIGEN__::__minus(v0, v2);

        __GEIGEN__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
        __GEIGEN__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
        __GEIGEN__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

        param[1].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
        param[1].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

        if (param[1].x > 0.0 && param[1].x < 1.0 && param[1].y >= 0.0) {
            return 4; // PE v2v3
        }
        else {
            basis0 = __GEIGEN__::__minus(v1, v3);
            basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
            basis2 = __GEIGEN__::__minus(v0, v3);

            __GEIGEN__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
            __GEIGEN__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
            __GEIGEN__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

            param[2].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
            param[2].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

            if (param[2].x > 0.0 && param[2].x < 1.0 && param[2].y >= 0.0) {
                return 5; // PE v3v1
            }
            else {
                if (param[0].x <= 0.0 && param[2].x >= 1.0) {
                    return 0; // PP v1
                }
                else if (param[1].x <= 0.0 && param[0].x >= 1.0) {
                    return 1; // PP v2
                }
                else if (param[2].x <= 0.0 && param[1].x >= 1.0) {
                    return 2; // PP v3
                }
                else {
                    return 6; // PT
                }
            }
        }
    }
}

__device__
int _dType_edge_edge(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 u = __GEIGEN__::__minus(v1, v0);
    double3 v = __GEIGEN__::__minus(v3, v2);
    double3 w = __GEIGEN__::__minus(v0, v2);

    double a = __GEIGEN__::__squaredNorm(u);
    double b = __GEIGEN__::__v_vec_dot(u, v);
    double c = __GEIGEN__::__squaredNorm(v);
    double d = __GEIGEN__::__v_vec_dot(u, w);
    double e = __GEIGEN__::__v_vec_dot(v, w);

    double D = a * c - b * b; // always >= 0
    double tD = D; // tc = tN / tD, default tD = D >= 0
    double sN, tN;
    int defaultCase = 8;
    sN = (b * e - c * d);
    if (sN <= 0.0) { // sc < 0 => the s=0 edge is visible
        tN = e;
        tD = c;
        defaultCase = 2;
    }
    else if (sN >= D) { // sc > 1  => the s=1 edge is visible
        tN = e + b;
        tD = c;
        defaultCase = 5;
    }
    else {
        tN = (a * e - b * d);
        if (tN > 0.0 && tN < tD && (__GEIGEN__::__v_vec_dot(w, __GEIGEN__::__v_vec_cross(u, v)) == 0.0 || __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(u, v)) < 1.0e-20 * a * c)) {
            if (sN < D / 2) {
                tN = e;
                tD = c;
                defaultCase = 2;
            }
            else {
                tN = e + b;
                tD = c;
                defaultCase = 5;
            }
        }
    }

    if (tN <= 0.0) {
        if (-d <= 0.0) {
            return 0;
        }
        else if (-d >= a) {
            return 3;
        }
        else {
            return 6;
        }
    }
    else if (tN >= tD) {
        if ((-d + b) <= 0.0) {
            return 1;
        }
        else if ((-d + b) >= a) {
            return 4;
        }
        else {
            return 7;
        }
    }

    return defaultCase;
}

__device__ __forceinline__
double point_point_distance(const double3& v0, const double3& v1)
{
    return __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1));
}

__device__ __forceinline__
double point_triangle_distance(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v2, v1), __GEIGEN__::__minus(v3, v1));
    //double3 test = __GEIGEN__::__minus(v0, v1);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v0, v1), b);//(v0 - v1).dot(b);
    //printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y, test.z, aTb);
    return aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__ __forceinline__
double point_edge_distance(const double3& v0, const double3& v1, const double3& v2)
{
    return __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0))) / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v1));
}

__device__ __forceinline__
double edge_edge_distance(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v3, v2));//(v1 - v0).cross(v3 - v2);
    //if(__GEIGEN__::__norm(b) <1e-6)
    //    b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)), __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    return aTb * aTb / __GEIGEN__::__squaredNorm(b);
}


__device__ __forceinline__
double _d_EEParallel(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)), __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    return aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__
double edge_edge_distance_unclassified(
    const double3& ea0,
    const double3& ea1,
    const double3& eb0,
    const double3& eb1)
{
    switch (_dType_edge_edge(ea0, ea1, eb0, eb1)) {
    case 0:
        return point_point_distance(ea0, eb0);
    case 1:
        return point_point_distance(ea0, eb1);
    case 2:
        return point_edge_distance(ea0, eb0, eb1);
    case 3:
        return point_point_distance(ea1, eb0);
    case 4:
        return point_point_distance(ea1, eb1);
    case 5:
        return point_edge_distance(ea1, eb0, eb1);
    case 6:
        return point_edge_distance(eb0, ea0, ea1);
    case 7:
        return point_edge_distance(eb1, ea0, ea1);
    case 8:
        return edge_edge_distance(ea0, ea1, eb0, eb1);
    default:
        return 1e32;
    }
}

__device__
double point_triangle_distance_unclassified(
    const double3& p,
    const double3& t0,
    const double3& t1,
    const double3& t2)
{
    switch (_dType_point_triangle(p, t0, t1, t2)) {
    case 0:
        return point_point_distance(p, t0);
    case 1:
        return point_point_distance(p, t1);
    case 2:
        return point_point_distance(p, t2);
    case 3:
        return point_edge_distance(p, t0, t1);
    case 4:
        return point_edge_distance(p, t1, t2);
    case 5:
        return point_edge_distance(p, t2, t0);
    case 6:
        return point_triangle_distance(p, t0, t1, t2);
    default:
        return 1e32;
    }
}

__device__
double edge_edge_ccd(
    const double3& _ea0,
    const double3& _ea1,
    const double3& _eb0,
    const double3& _eb1,
    const double3& _dea0,
    const double3& _dea1,
    const double3& _deb0,
    const double3& _deb1,
    double eta, double thickness)
{
    double3 ea0 = _ea0, ea1 = _ea1, eb0 = _eb0, eb1 = _eb1, dea0 = _dea0, dea1 = _dea1, deb0 = _deb0, deb1 = _deb1;
    double3 temp0 = __GEIGEN__::__add(dea0, dea1);
    double3 temp1 = __GEIGEN__::__add(deb0, deb1);
    double3 mov = __GEIGEN__::__s_vec_multiply(__GEIGEN__::__add(temp0, temp1), -0.25);

    dea0 = __GEIGEN__::__add(dea0, mov);
    dea1 = __GEIGEN__::__add(dea1, mov);
    deb0 = __GEIGEN__::__add(deb0, mov);
    deb1 = __GEIGEN__::__add(deb1, mov);

    double max_disp_mag = sqrt(__m_max(__GEIGEN__::__squaredNorm(dea0), __GEIGEN__::__squaredNorm(dea1))) + sqrt(__m_max(__GEIGEN__::__squaredNorm(deb0), __GEIGEN__::__squaredNorm(deb1)));
    if (max_disp_mag == 0)
        return 1.0;

    double dist2_cur = edge_edge_distance_unclassified(ea0, ea1, eb0, eb1);

    double dFunc = dist2_cur - thickness * thickness;
    if (dFunc <= 0) {
        double dists0 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea0, eb0));
        double dists1 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea0, eb1));
        double dists2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea1, eb0));
        double dists3 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea1, eb1));

        dist2_cur = __m_min(__m_min(dists0, dists1), __m_min(dists2, dists3));
        dFunc = dist2_cur - thickness * thickness;
    }
    double dist_cur = sqrt(dist2_cur);
    double gap = eta * dFunc / (dist_cur + thickness);
    double toc = 0.0;
    int count = 0;
    while (true) {
        count++;
        //if (count > 1000) return 1;
        if (count > 5000)
            printf("ee  %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n", _dea0.x, _dea0.y, _dea0.z, _dea1.x, _dea1.y, _dea1.z,
                _deb0.x, _deb0.y, _deb0.z, _deb1.x, _deb1.y, _deb1.z, _ea0.x, _ea0.y, _ea0.z, _ea1.x, _ea1.y, _ea1.z, _eb0.x, _eb0.y, _eb0.z, _eb1.x, _eb1.y, _eb1.z);
        double toc_lower_bound = (1 - eta) * dFunc / ((dist_cur + thickness) * max_disp_mag);
        ea0 = __GEIGEN__::__add(ea0, __GEIGEN__::__s_vec_multiply(dea0, toc_lower_bound));
        ea1 = __GEIGEN__::__add(ea1, __GEIGEN__::__s_vec_multiply(dea1, toc_lower_bound));
        eb0 = __GEIGEN__::__add(eb0, __GEIGEN__::__s_vec_multiply(deb0, toc_lower_bound));
        eb1 = __GEIGEN__::__add(eb1, __GEIGEN__::__s_vec_multiply(deb1, toc_lower_bound));

        dist2_cur = edge_edge_distance_unclassified(ea0, ea1, eb0, eb1);
        dFunc = dist2_cur - thickness * thickness;
        if (dFunc <= 0) {
            double dists0 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea0, eb0));
            double dists1 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea0, eb1));
            double dists2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea1, eb0));
            double dists3 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(ea1, eb1));
            
            dist2_cur = __m_min(__m_min(dists0, dists1), __m_min(dists2, dists3));
            dFunc = dist2_cur - thickness * thickness;
        }
        dist_cur = sqrt(dist2_cur);
        if (toc && (dFunc / (dist_cur + thickness) < gap)) {
            break;
        }
        toc += toc_lower_bound;
        if (toc > 1.0)
            return 1.0;
    }
    return toc;
}

__device__
double point_triangle_ccd(
    const double3& _p,
    const double3& _t0,
    const double3& _t1,
    const double3& _t2,
    const double3& _dp,
    const double3& _dt0,
    const double3& _dt1,
    const double3& _dt2,
    double eta, double thickness)
{
    double3 p = _p, t0 = _t0, t1 = _t1, t2 = _t2, dp = _dp, dt0 = _dt0, dt1 = _dt1, dt2 = _dt2;

    double3 temp0 = __GEIGEN__::__add(dt0, dt1);
    double3 temp1 = __GEIGEN__::__add(dt2, dp);
    double3 mov = __GEIGEN__::__s_vec_multiply(__GEIGEN__::__add(temp0, temp1), -0.25);

    dt0 = __GEIGEN__::__add(dt0, mov);
    dt1 = __GEIGEN__::__add(dt1, mov);
    dt2 = __GEIGEN__::__add(dt2, mov);
    dp = __GEIGEN__::__add(dp, mov);

    double disp_mag2_vec0 = __GEIGEN__::__squaredNorm(dt0);
    double disp_mag2_vec1 = __GEIGEN__::__squaredNorm(dt1);
    double disp_mag2_vec2 = __GEIGEN__::__squaredNorm(dt2);

    double max_disp_mag = __GEIGEN__::__norm(dp) + sqrt(__m_max(disp_mag2_vec0, __m_max(disp_mag2_vec1, disp_mag2_vec2)));
    if (max_disp_mag == 0)
        return 1.0;

    double dist2_cur = point_triangle_distance_unclassified(p, t0, t1, t2);
    double dist_cur = sqrt(dist2_cur);
    double gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    double toc = 0.0;
    int count = 0;
    while (true) {
        count++;
        //if (count > 1000) return 1;
        if (count > 5000)
            printf("pt  %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n", _dp.x, _dp.y, _dp.z, _dt0.x, _dt0.y, _dt0.z,
                _dt1.x, _dt1.y, _dt1.z, _dt2.x, _dt2.y, _dt2.z, _p.x, _p.y, _p.z, _t0.x, _t0.y, _t0.z, _t1.x, _t1.y, _t1.z, _t2.x, _t2.y, _t2.z);
        double toc_lower_bound = (1 - eta) * (dist2_cur - thickness * thickness) / ((dist_cur + thickness) * max_disp_mag);

        p = __GEIGEN__::__add(p, __GEIGEN__::__s_vec_multiply(dp, toc_lower_bound));
        t0 = __GEIGEN__::__add(t0, __GEIGEN__::__s_vec_multiply(dt0, toc_lower_bound));
        t1 = __GEIGEN__::__add(t1, __GEIGEN__::__s_vec_multiply(dt1, toc_lower_bound));
        t2 = __GEIGEN__::__add(t2, __GEIGEN__::__s_vec_multiply(dt2, toc_lower_bound));

        dist2_cur = point_triangle_distance_unclassified(p, t0, t1, t2);
        dist_cur = sqrt(dist2_cur);
        if (toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)) {
            break;
        }

        toc += toc_lower_bound;
        if (toc > 1.0) {
            return 1.0;
        }
    }
    return toc;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


typedef struct {
    double3 ad, bd, cd, pd;
    double3 a0, b0, c0, p0;
} NewtonCheckData;

inline __device__ bool _insideTriangle(double3 a, double3 b, double3 c, double3 p)
{
    double3 n, da, db, dc;
    double wa, wb, wc;

    double3 ba = __GEIGEN__::__minus(b, a);
    double3 ca = __GEIGEN__::__minus(c, a);

    n = __GEIGEN__::__v_vec_cross(ba, ca);//cross(ba, ca);

    da = __GEIGEN__::__minus(a, p);
    db = __GEIGEN__::__minus(b, p);
    dc = __GEIGEN__::__minus(c, p);
    //da = a - p, db = b - p, dc = c - p;
    if ((wa = __GEIGEN__::__v_vec_dot(__GEIGEN__::__v_vec_cross(db, dc), n)) < 0.0f) return false;
    if ((wb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__v_vec_cross(dc, da), n)) < 0.0f) return false;
    if ((wc = __GEIGEN__::__v_vec_dot(__GEIGEN__::__v_vec_cross(da, db), n)) < 0.0f) return false;

    //Compute barycentric coordinates
    double area2 = __GEIGEN__::__v_vec_dot(n, n);
    wa /= area2, wb /= area2, wc /= area2;


    return true;
}


inline __device__ int solveQuadric(double c[3], double s[2], const double& errorRate)
{
    double p, q, D;

    // make sure we have a d2 equation

    if (c[2] < errorRate) {
        if ((c[1]) < errorRate)
            return 0;
        s[0] = -c[0] / c[1];
        return 1;
    }

    // normal for: x^2 + px + q
    p = c[1] / (2.0f * c[2]);
    q = c[0] / c[2];
    D = p * p - q;

    if ((D)< errorRate)
    {
        // one float root
        s[0] = s[1] = -p;
        return 1;
    }

    if (D < 0.0f)
        // no real root
        return 0;

    else
    {
        // two real roots
        double sqrt_D = sqrt(D);
        s[0] = sqrt_D - p;
        s[1] = -sqrt_D - p;
        return 2;
    }
}
inline __device__ int solveCubic(double c[4], double s[3], const double& errorRate)
{
    int	i, num;
    double	sub,
        A, B, C,
        sq_A, p, q,
        cb_p, D;

    if (c[3]< errorRate) {
        return solveQuadric(c, s, errorRate);
    }

    A = c[2] / c[3];
    B = c[1] / c[3];
    C = c[0] / c[3];

    sq_A = A * A;
    double ONE_DIV_3 = 1.0 / 3;
    p = ONE_DIV_3 * (-ONE_DIV_3 * sq_A + B);
    q = 0.5f * (2.0f / 27.0f * A * sq_A - ONE_DIV_3 * A * B + C);

    // use Cardano's formula

    cb_p = p * p * p;
    D = q * q + cb_p;
    double _PI = 3.1415926535897932;
    if ((D)< errorRate)
    {
        if ((q)< errorRate)
        {
            // one triple solution
            s[0] = 0.0f;
            num = 1;
        }
        else
        {
            // one single and one float solution
            double u = cbrt(-q);
            s[0] = 2.0f * u;
            s[1] = -u;
            num = 2;
        }
    }
    else
        if (D < 0.0f)
        {
            // casus irreductibilis: three real solutions
            double phi = ONE_DIV_3 * acos(-q / sqrt(-cb_p));
            double t = 2.0f * sqrt(-p);
            s[0] = t * cos(phi);
            s[1] = -t * cos(phi + _PI / 3.0f);
            s[2] = -t * cos(phi - _PI / 3.0f);
            num = 3;
        }
        else
        {
            // one real solution
            double sqrt_D = sqrt(D);
            double u = cbrt(sqrt_D + fabs(q));
            if (q > 0.0f)
                s[0] = -u + p / u;
            else
                s[0] = u - p / u;
            num = 1;
        }

    // resubstitute
    sub = ONE_DIV_3 * A;
    for (i = 0; i < num; i++)
        s[i] -= sub;
    return num;
}

inline __device__ void _equateCubic_VF(
    double3 a0, double3 ad, double3 b0, double3 bd,
    double3 c0, double3 cd, double3 p0, double3 pd,
    double& a, double& b, double& c, double& d, const double& thickness)
{
    double3 dab, dac, dap;
    double3 oab, oac, oap;
    double3 dabXdac, dabXoac, oabXdac, oabXoac;
    
    dab =__GEIGEN__::__minus(bd, ad), dac =__GEIGEN__::__minus(cd , ad), dap =__GEIGEN__::__minus(pd , ad);
    oab =__GEIGEN__::__minus(b0, a0), oac =__GEIGEN__::__minus(c0 , a0), oap =__GEIGEN__::__minus(p0 , a0);
    
    dabXdac = __GEIGEN__::__v_vec_cross(dab, dac);
    dabXoac = __GEIGEN__::__v_vec_cross(dab, oac);
    oabXdac = __GEIGEN__::__v_vec_cross(oab, dac);
    oabXoac = __GEIGEN__::__v_vec_cross(oab, oac);
    
    a = __GEIGEN__::__v_vec_dot(dap, dabXdac);
    b = __GEIGEN__::__v_vec_dot(oap, dabXdac) + __GEIGEN__::__v_vec_dot(dap,__GEIGEN__::__add(dabXoac , oabXdac));
    c = __GEIGEN__::__v_vec_dot(dap, oabXoac) + __GEIGEN__::__v_vec_dot(oap,__GEIGEN__::__add(dabXoac , oabXdac));
    d = thickness * __GEIGEN__::__v_vec_dot(oap, oabXoac);
}

inline __device__ double IntersectVF(double3 ta0, double3 tb0, double3 tc0,
    double3 ad, double3 bd, double3 cd,
    double3 q0, double3 qd,
    const double& errorRate, const double& thickness)
{
    double collisionTime = 1.0;

    double a, b, c, d; /* cubic polynomial coefficients */
    _equateCubic_VF(ta0, ad, tb0, bd, tc0, cd, q0, qd, a, b, c, d, thickness);

    //if ((a) < errorRate && (b) < errorRate && (c) < errorRate && (d) < errorRate)
    //    return 1.0;

    double roots[3];
    double coeffs[4];
    coeffs[3] = a, coeffs[2] = b, coeffs[1] = c, coeffs[0] = d;
    int num;// = solveCubic(coeffs, roots, errorRate);

    __GEIGEN__::__NewtonSolverForCubicEquation(a, b, c, d, roots, num, errorRate);

    if (num == 0)
        return 1.0;

    for (int i = 0; i < num; i++) {
        double r = roots[i];
        if (r < 0 || r > 1) continue;

        if (_insideTriangle(
            __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(ad, r), ta0),
            __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(bd, r), tb0),
            __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(cd, r), tc0),
            __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(qd, r), q0))) {
            if (collisionTime > r) {
                collisionTime = r;
            }
        }
    }

    return collisionTime;
}

__device__ 
double doCCDVF(const double3& _p,
    const double3& _t0,
    const double3& _t1,
    const double3& _t2,
    const double3& _dp,
    const double3& _dt0,
    const double3& _dt1,
    const double3& _dt2,
    double errorRate, double thickness)
{
    double ret = IntersectVF(_t0, _t1, _t2, _dt0, _dt1, _dt2, _p, _dp, errorRate, thickness);

    return ret;
}