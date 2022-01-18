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


struct PrimitiveCalcNormal : zeno::INode {

  // for parallel atomic float add
  template <typename DstT, typename SrcT> constexpr auto reinterpret_bits(SrcT &&val) {
    using Src = std::remove_cv_t<std::remove_reference_t<SrcT>>;
    using Dst = std::remove_cv_t<std::remove_reference_t<DstT>>;
    static_assert(sizeof(Src) == sizeof(Dst),
                  "Source Type and Destination Type must be of the same size");
    return reinterpret_cast<Dst const volatile &>(val);
  }

  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");

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
        auto n = zeno::cross(pos[ind[1]] - pos[ind[0]], pos[ind[2]] - pos[ind[0]]);

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

#if defined(_OPENMP) && defined(__GNUG__)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < nrm.size(); i++) {
        nrm[i] = -zeno::normalize(nrm[i]);
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveCalcNormal, {
    {"prim"},
    {"prim"},
    {},
    {"primitive"},
});

struct PrimitiveSplitEdges : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");

    prim->foreach_attr([&] (auto &, auto &arr) {
        auto oldarr = arr;
        arr.resize(prim->tris.size() * 3);
        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto ind = prim->tris[i];
            arr[i * 3 + 0] = oldarr[ind[0]];
            arr[i * 3 + 1] = oldarr[ind[1]];
            arr[i * 3 + 2] = oldarr[ind[2]];
        }
    });
    prim->resize(prim->tris.size() * 3);

    for (size_t i = 0; i < prim->tris.size(); i++) {
        prim->tris[i] = zeno::vec3i(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveSplitEdges, {
    {"prim"},
    {"prim"},
    {},
    {"primitive"},
});


struct PrimitiveFaceToEdges : zeno::INode {
  std::pair<int, int> sorted(int x, int y) {
      return x < y ? std::make_pair(x, y) : std::make_pair(y, x);
  }

  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    std::set<std::pair<int, int>> lines;

    for (int i = 0; i < prim->tris.size(); i++) {
        auto uvw = prim->tris[i];
        int u = uvw[0], v = uvw[1], w = uvw[2];
        lines.insert(sorted(u, v));
        lines.insert(sorted(v, w));
        lines.insert(sorted(u, w));
    }
    for (auto [u, v]: lines) {
        prim->lines.emplace_back(u, v);
    }

    if (get_param<bool>("clearFaces")) {
        prim->tris.clear();
    }
    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveFaceToEdges,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"bool", "clearFaces", "1"},
    }, /* category: */ {
    "primitive",
    }});

}
