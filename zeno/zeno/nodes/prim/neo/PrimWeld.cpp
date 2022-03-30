#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <unordered_map>

namespace zeno {
namespace {

struct PrimWeld : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        auto isAverage = get_input<StringObject>("method")->get() == "average";

        std::unordered_multimap<int, int> lut;
        auto &tag = prim->verts.attr<int>(tagAttr);
        for (int i = 0; i < prim->size(); i++) {
            lut.insert({tag[i], i});
        }
        std::vector<int> revamp;
        std::vector<int> unrevamp(prim->size(), -1);
        revamp.resize(lut.size());
        int nrevamp = 0;
        for (auto it = lut.begin(); it != lut.end();) {
            auto nit = std::find_if(std::next(it), lut.end(), [val = it->first] (auto const &p) {
                return p.first != val;
            });
            auto start = it->second;
            if (isAverage) {
                vec3f average = prim->verts[start];
                int count = 1;
                for (++it; it != nit; ++it) {
                    unrevamp[it->second] = nrevamp;
                    auto pos = prim->verts[it->second];
                    average += pos;
                    ++count;
                }
                average *= 1.f / count;
                prim->verts[start] = average;
            } else {
                for (; it != nit; ++it) {
                    //printf("(%d) %d -> %d\n", it->first, it->second, nrevamp);
                    unrevamp[it->second] = nrevamp;
                }
            }
            revamp[nrevamp] = start;
            ++nrevamp;
        }
        revamp.resize(nrevamp);
        primRevampVerts(prim.get(), revamp, &unrevamp);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimWeld, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "tagAttr", "tag"},
    {"enum oneof average", "method", "oneof"},
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
