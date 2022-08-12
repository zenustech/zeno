#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/para/parallel_push_back.h>
#include <zeno/utils/vec.h>
#include <algorithm>

namespace zeno {
namespace {

struct PrimConnectTape : INode {
    virtual void apply() override {
        auto prim1 = get_input<PrimitiveObject>("prim1");
        auto prim2 = get_input<PrimitiveObject>("prim2");
        auto faceType = get_input2<std::string>("faceType");
        auto isCloseRing = get_input2<bool>("isCloseRing");

        auto prim = std::make_shared<PrimitiveObject>();
        prim->verts.resize(prim1->verts.size() + prim2->verts.size());

        prim1->verts.forall_attr([&] (auto const &key, auto &arr1) {
            using T = std::decay_t<decltype(arr1[0])>;
            prim->add_attr<T>(key);
        });
        prim2->verts.forall_attr([&] (auto const &key, auto &arr2) {
            using T = std::decay_t<decltype(arr2[0])>;
            prim->add_attr<T>(key);
        });

        int n1 = prim1->verts.size();
        int n2 = prim2->verts.size();
        prim->verts.forall_attr([&] (auto const &key, auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            if (prim1->has_attr(key)) {
                auto &arr1 = prim1->attr<T>(key);
                std::copy(arr1.begin(), arr1.end(), arr.begin());
            }
            if (prim2->has_attr(key)) {
                auto &arr2 = prim2->attr<T>(key);
                std::copy(arr2.begin(), arr2.end(), arr.begin() + n1);
            }
        });

        int n = std::max(n1, n2);
        auto p1 = [&] (int i) {
            return std::min(i, n1);
        };
        auto p2 = [&] (int i) {
            return std::min(i, n2) + n1;
        };

        if (faceType == "lines") {
            prim->lines.resize(n);
            for (int i = 0; i < n; i++) {
                prim->lines[i] = {p1(i), p2(i)};
            }
        } else if (faceType == "quads") {
            prim->quads.resize(n - (int)!isCloseRing);
            for (int i = 0; i < n - 1; i++) {
                prim->quads[i] = {p1(i), p2(i), p2(i + 1), p1(i + 1)};
            }
            if (isCloseRing) {
                prim->quads[n - 1] = {p1(n - 1), p2(n - 1), p2(0), p1(0)};
            }
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimConnectTape, {
    {
    {"PrimitiveObject", "prim1"},
    {"PrimitiveObject", "prim2"},
    {"enum quads lines none", "faceType", "quads"},
    {"bool", "isCloseRing", "0"},
    {"string", "edgeMaskAttr", ""},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimConnectBridge : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto faceType = get_input2<std::string>("faceType");
        auto edgeIndAttr = get_input2<std::string>("edgeIndAttr");

        auto &ind = prim->lines.attr<int>(edgeIndAttr);
#ifdef ZENO_PARALLEL_STL
        parallel_push_back(prim->quads, prim->lines.size(), [&] (size_t i, auto &quads) {
            int j = ind[i];
            if (j != -1) {
                auto l1 = prim->lines[i];
                auto l2 = prim->lines[j];
                vec4i quad(l1[0], l1[1], l2[1], l2[0]);
                quads.push_back(quad);
            }
        });
#else
        for (int i = 0; i < prim->lines.size(); i++) {
            int j = ind[i];
            if (j == -1) continue;
            auto l1 = prim->lines[i];
            auto l2 = prim->lines[j];
            vec4i quad(l1[0], l1[1], l2[1], l2[0]);
            prim->quads.push_back(quad);
        }
#endif
    }
};

ZENDEFNODE(PrimConnectBridge, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "edgeIndAttr", "tag"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
