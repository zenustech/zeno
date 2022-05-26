#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#if defined(_OPENMP) && defined(__GNUG__)
#include <omp.h>
#endif

namespace zeno {
// for parallel atomic float add
template <typename DstT, typename SrcT> constexpr auto reinterpret_bits(SrcT &&val) {
    using Src = std::remove_cv_t<std::remove_reference_t<SrcT>>;
    using Dst = std::remove_cv_t<std::remove_reference_t<DstT>>;
    static_assert(sizeof(Src) == sizeof(Dst),
                  "Source Type and Destination Type must be of the same size");
    return reinterpret_cast<Dst const volatile &>(val);
  }
ZENO_API void primCalcNormal(zeno::PrimitiveObject* prim, float flip)
{
        auto &nrm = prim->add_attr<zeno::vec3f>("nrm");
    auto &pos = prim->attr<zeno::vec3f>("pos");

#if defined(_OPENMP) && defined(__GNUG__)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < nrm.size(); i++) {
        nrm[i] = zeno::vec3f(0);
    }

#if defined(_OPENMP) && defined(__GNUG__)
    auto atomicCas = [](int* dest, int expected, int desired) {
        __atomic_compare_exchange_n(const_cast<int volatile *>(dest), &expected,
                                    desired, false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_RELAXED);
        return expected;
    };
    auto atomicFloatAdd = [&](float*dst, float val) {
        static_assert(sizeof(float) == sizeof(int), "sizeof float != sizeof int");
        int oldVal = reinterpret_bits<int>(*dst);
        int newVal = reinterpret_bits<int>(reinterpret_bits<float>(oldVal) + val), readVal{};
        while ((readVal = atomicCas((int*)dst, oldVal, newVal)) != oldVal) {
            oldVal = readVal;
            newVal = reinterpret_bits<int>(reinterpret_bits<float>(readVal) + val);
        }
        return reinterpret_bits<float>(oldVal);
    };
#endif

#if defined(_OPENMP) && defined(__GNUG__)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < prim->tris.size(); i++) {
        auto ind = prim->tris[i];
        auto n = cross(pos[ind[1]] - pos[ind[0]], pos[ind[2]] - pos[ind[0]]);

#if defined(_OPENMP) && defined(__GNUG__)
        for (int j = 0; j != 3; ++j) {
            auto &n_i = nrm[ind[j]];
            for (int d = 0; d != 3; ++d)
                atomicFloatAdd(&n_i[d], n[d]);
        }
#else
        nrm[ind[0]] += n;
        nrm[ind[1]] += n;
        nrm[ind[2]] += n;
#endif
    }

    constexpr auto eps = std::numeric_limits<float>::epsilon();
#if defined(_OPENMP) && defined(__GNUG__)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < nrm.size(); i++) {
        auto n = nrm[i];
        if (std::abs(n[0]) < eps || std::abs(n[1]) < eps || std::abs(n[2]) < eps) continue;
        nrm[i] = flip * normalize(nrm[i]);
    }
}
struct PrimitiveCalcNormal : zeno::INode {

  

  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto flip = get_input<NumericObject>("flip")->get<int>() == 1? -1.0f:1.0f;
    primCalcNormal(prim.get(), flip);

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveCalcNormal, {
    {"prim",
    {"int", "flip", "0"}
    },
    {"prim"},
    {},
    {"primitive"},
});

}
