#pragma once

#include <x86intrin.h>

// gcc -march=native -dM -E - < /dev/null | egrep 'SSE|AVX|MMX'
#ifdef __SSE__
#define HG_SIMD_FLOAT4
#endif
#ifdef __AVX__
#define HG_SIMD_FLOAT8
#endif
#ifdef __AVX512F__
#define HG_SIMD_FLOAT16
#endif

namespace hg::simd {

  template <class T, int N> struct _SIMD {};

  template <class T, int N> struct SIMD {
    typedef _SIMD<T, N> M;

    M m;
    SIMD() {}
    SIMD(M const &m) : m(m) {}

    explicit SIMD(T x) { m.setall(x); }

    SIMD(T x, T y, T z, T w) { m.set(x, y, z, w); }

    SIMD(T x, T y, T z, T w, T a, T b, T c, T d) { m.set(x, y, z, w, a, b, c, d); }

    SIMD(T x, T y, T z, T w, T a, T b, T c, T d,
         T e, T f, T g, T h, T i, T j, T k, T l) {
         m.set(x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l);
    }

#define _DEF_IOP2(iop, name) \
  SIMD &iop(SIMD const &o) {       \
    m.name##p(m, o.m);       \
    return *this;            \
  }                          \
                             \
  SIMD &iop(T o) {              \
    m.name##p(m, SIMD(o).m);    \
    return *this;            \
  }

#define _DEF_OP2(op, name)       \
  SIMD op(SIMD const &o) const {       \
    M r;                         \
    r.name##p(m, o.m);           \
    return r;                    \
  }                              \
                                 \
  SIMD op(T o) const {              \
    M r;                         \
    r.name##p(m, SIMD(o).m);        \
    return r;                    \
  }                              \
                                 \
  friend SIMD op(T t, SIMD const &o) { \
    M r;                         \
    r.name##p(SIMD(t).m, o.m);      \
    return r;                    \
  }

#define _DEF_OP2N(name)   \
  _DEF_OP2(i##name, name) \
  _DEF_OP2(name, name)

#define _DEF_OP2I(op, name)       \
  _DEF_IOP2(operator op##=, name) \
  _DEF_OP2(operator op, name)

#define _DEF_OP2C(op, name) _DEF_OP2(operator op, name)

    _DEF_OP2I(+, add);
    _DEF_OP2I(-, sub);
    _DEF_OP2I(*, mul);
    _DEF_OP2I(/, div);
    _DEF_OP2I(|, or);
    _DEF_OP2I(&, and);
    _DEF_OP2I(^, xor);
    _DEF_OP2C(==, cmpeq);
    _DEF_OP2C(!=, cmpneq);
    _DEF_OP2C(<, cmplt);
    _DEF_OP2C(<=, cmple);
    _DEF_OP2C(>, cmpgt);
    _DEF_OP2C(>=, cmpge);
    _DEF_OP2N(andnot);
    _DEF_OP2N(max);
    _DEF_OP2N(min);

    SIMD operator~() const {
      M r;
      r.setall(0);
      r.andnot(m, r);
    }

    int mask() const { return m.movemask(); }

#undef _DEF_OP2
#undef _DEF_IOP2
#undef _DEF_OP2I
#undef _DEF_OP2C
#undef _DEF_OP2N

#define _DEF_OP1(name) \
  SIMD name() const {     \
    M r;               \
    r.name##p(m);      \
    return r;          \
  }

    _DEF_OP1(rcp);
    _DEF_OP1(sqrt);
    _DEF_OP1(rsqrt);

#undef _DEF_OP1

    T operator[](int i) const {
      float a[N];
      m.store(a);
      return a[i];
    }

    void load(T const *p) { m.load(p); }

    void store(T *p) const { m.store(p); }

    void loadu(T const *p) { m.loadu(p); }

    void storeu(T *p) const { m.storeu(p); }

    static SIMD fromu(T const &p) {
      SIMD v;
      v.loadu(&p);
      return v;
    }

    static SIMD from(T const &p) {
      SIMD v;
      v.load(&p);
      return v;
    }

    struct _Assign {
      T *p;
      _Assign &operator=(SIMD const &v) {
        v.store(p);
        return *this;
      }
    };

    struct _AssignU {
      T *p;
      _AssignU &operator=(SIMD const &v) {
        v.storeu(p);
        return *this;
      }
    };

    static _Assign to(T &p) { return {&p}; }

    static _AssignU tou(T &p) { return {&p}; }
  };

  typedef SIMD<float, 4> float4;
  typedef SIMD<float, 8> float8;
  typedef SIMD<float, 16> float16;

}  // namespace hg::SIMD
