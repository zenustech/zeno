#pragma once

#include "simd.hpp"

namespace hg::simd {

  template <> struct _SIMD<float, 4> {
    __m128 m;

    _SIMD() {}
    explicit _SIMD(__m128 m) : m(m) {}

    void sets(float x) { m = _mm_set_ss(x); }

    void setall(float x) { m = _mm_set1_ps(x); }

    void set(float x, float y, float z, float w) { m = _mm_set_ps(w, z, y, x); }

    void load(float const *p) { m = _mm_load_ps(p); }

    void store(float *p) const { _mm_store_ps(p, m); }

    void loadu(float const *p) { m = _mm_loadu_ps(p); }

    void storeu(float *p) const { _mm_storeu_ps(p, m); }

    template <int x, int y, int z, int w> void shuffle(_SIMD const &l, _SIMD const &r) {
      m = _mm_shuffle_ps(l.m, r.m, _MM_SHUFFLE(w, z, y, x));
    }

    void unpackhi(_SIMD const &l, _SIMD const &r) { m = _mm_unpackhi_ps(l.m, r.m); }

    void unpacklo(_SIMD const &l, _SIMD const &r) { m = _mm_unpacklo_ps(l.m, r.m); }

    void movelh(_SIMD const &l, _SIMD const &r) { m = _mm_movelh_ps(l.m, r.m); }

    void movehl(_SIMD const &l, _SIMD const &r) { m = _mm_movehl_ps(l.m, r.m); }

    int movemask() const { return _mm_movemask_ps(m); }

    float gets() const { return _mm_cvtss_f32(m); }

    void dotp(_SIMD const &l, _SIMD const &r, int imm) { m = _mm_dp_ps(l.m, r.m, imm); }

    void blend(_SIMD const &l, _SIMD const &r, int imm) { m = _mm_blend_ps(l.m, r.m, imm); }

    void blendv(_SIMD const &l, _SIMD const &r, _SIMD const &c) { m = _mm_blendv_ps(l.m, r.m, c.m); }

#define _DEF_OP2P(x) \
  void x##p(_SIMD const &l, _SIMD const &r) { m = _mm_##x##_ps(l.m, r.m); }
#define _DEF_OP2S(x) \
  void x##s(_SIMD const &l, _SIMD const &r) { m = _mm_##x##_ss(l.m, r.m); }

#define _DEF_OP2(x) \
  _DEF_OP2P(x)      \
  _DEF_OP2S(x)

    _DEF_OP2(add);
    _DEF_OP2(sub);
    _DEF_OP2P(hadd);
    _DEF_OP2P(hsub);
    _DEF_OP2P(addsub);
    _DEF_OP2(mul);
    _DEF_OP2(div);
    _DEF_OP2(min);
    _DEF_OP2(max);
    _DEF_OP2P(or);
    _DEF_OP2P(and);
    _DEF_OP2P(xor);
    _DEF_OP2P(andnot);
    _DEF_OP2(cmpeq);
    _DEF_OP2(cmplt);
    _DEF_OP2(cmple);
    _DEF_OP2(cmpgt);
    _DEF_OP2(cmpge);
    _DEF_OP2(cmpneq);
    _DEF_OP2(cmpnlt);
    _DEF_OP2(cmpnle);
    _DEF_OP2(cmpngt);
    _DEF_OP2(cmpnge);
    _DEF_OP2S(move);

#undef _DEF_OP2
#undef _DEF_OP2P
#undef _DEF_OP2S

#define _DEF_OP1(x)                                 \
  void x##p(_SIMD const &o) { m = _mm_##x##_ps(o.m); } \
                                                    \
  void x##s(_SIMD const &o) { m = _mm_##x##_ss(o.m); }

    _DEF_OP1(rcp);
    _DEF_OP1(sqrt);
    _DEF_OP1(rsqrt);

#undef _DEF_OP1
  };

}  // namespace hg::SIMD
