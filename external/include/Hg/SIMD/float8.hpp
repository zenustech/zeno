#pragma once

#include "simd.hpp"

namespace hg::simd {

  template <> struct _SIMD<float, 8> {
    __m256 m;

    _SIMD() {}
    explicit _SIMD(__m256 m) : m(m) {}

    void setall(float x) { m = _mm256_set1_ps(x); }

    void set(float x, float y, float z, float w, float a, float b, float c, float d) {
      m = _mm256_set_ps(d, c, b, a, w, z, y, x);
    }

    void load(float const *p) { m = _mm256_load_ps(p); }

    void store(float *p) const { _mm256_store_ps(p, m); }

    void loadu(float const *p) { m = _mm256_loadu_ps(p); }

    void storeu(float *p) const { _mm256_storeu_ps(p, m); }

    template <int x, int y, int z, int w> void shuffle(_SIMD const &l, _SIMD const &r) {
      m = _mm256_shuffle_ps(l.m, r.m, _MM_SHUFFLE(w, z, y, x));
    }

    void unpackhi(_SIMD const &l, _SIMD const &r) { m = _mm256_unpackhi_ps(l.m, r.m); }

    void unpacklo(_SIMD const &l, _SIMD const &r) { m = _mm256_unpacklo_ps(l.m, r.m); }

    int movemask() const { return _mm256_movemask_ps(m); }

    float gets() const { return _mm256_cvtss_f32(m); }

    void dotp(_SIMD const &l, _SIMD const &r, int imm) { m = _mm256_dp_ps(l.m, r.m, imm); }

    void blend(_SIMD const &l, _SIMD const &r, int imm) { m = _mm256_blend_ps(l.m, r.m, imm); }

    void blendv(_SIMD const &l, _SIMD const &r, _SIMD const &c) { m = _mm256_blendv_ps(l.m, r.m, c.m); }

#define _DEF_OP2P(x) \
  void x##p(_SIMD const &l, _SIMD const &r) { m = _mm256_##x##_ps(l.m, r.m); }
#define _DEF_OP2S(x) \
  void x##s(_SIMD const &l, _SIMD const &r) { m = _mm256_##x##_ss(l.m, r.m); }

#define _DEF_OP2(x) \
  _DEF_OP2P(x)      \
  _DEF_OP2S(x)

    _DEF_OP2P(add);
    _DEF_OP2P(sub);
    _DEF_OP2P(hadd);
    _DEF_OP2P(hsub);
    _DEF_OP2P(addsub);
    _DEF_OP2P(mul);
    _DEF_OP2P(div);
    _DEF_OP2P(min);
    _DEF_OP2P(max);
    _DEF_OP2P(or);
    _DEF_OP2P(and);
    _DEF_OP2P(xor);
    _DEF_OP2P(andnot);

#undef _DEF_OP2
#undef _DEF_OP2P
#undef _DEF_OP2S

#define _DEF_OP1(x) \
  void x##p(_SIMD const &o) { m = _mm256_##x##_ps(o.m); }

    _DEF_OP1(rcp);
    _DEF_OP1(sqrt);
    _DEF_OP1(rsqrt);

#undef _DEF_OP1
  };

}  // namespace hg::SIMD
