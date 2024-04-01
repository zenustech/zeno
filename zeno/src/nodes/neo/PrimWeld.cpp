#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <unordered_map>
#include <algorithm>

namespace zeno {
namespace {

template <class T>
static void revamp_vector(std::vector<T> &arr, std::vector<int> const &revamp) {
    std::vector<T> newarr(arr.size());
    for (int i = 0; i < revamp.size(); i++) {
        newarr[i] = arr[revamp[i]];
    }
    std::swap(arr, newarr);
}

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
        std::vector<int> unrevamp(prim->size());
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
                average *= 1 / (float)count;
                prim->verts[start] = average;
            } else {
                for (; it != nit; ++it) {
                    //printf("(%d) %d -> %d\n", it->first, it->second, nrevamp);
                    unrevamp[it->second] = nrevamp;
                    // unrevamp[old_coor] = new_coor
                }
            }
            revamp[nrevamp] = start;
            ++nrevamp;
        }
        revamp.resize(nrevamp);
        //primRevampVerts(prim.get(), revamp, &unrevamp);

        revamp_vector(prim->verts.values, revamp);
        if (isAverage) {
            prim->verts.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                std::vector<T> new_arr(nrevamp);
                for (size_t i = 0; i < arr.size(); i++) {
                    new_arr[unrevamp[i]] += arr[i] / (T)lut.count(i);
                }
                arr = std::move(new_arr);
            });
        } else {
            prim->verts.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto &arr) {
                revamp_vector(arr, revamp);
            });
        }

        auto repair = [&] (int &x) {
            //printf("%d -> %d\n", x, unrevamp[x]);
            if (x >= 0 && x < unrevamp.size())
                x = unrevamp[x];
        };

        for (size_t i = 0; i < prim->points.size(); i++) {
            auto &ind = prim->points[i];
            repair(ind);
        }

        for (size_t i = 0; i < prim->lines.size(); i++) {
            auto &ind = prim->lines[i];
            repair(ind[0]);
            repair(ind[1]);
        }
        prim->lines->erase(std::remove_if(prim->lines.begin(), prim->lines.end(), [&] (auto const &ind) {
            return ind[0] == ind[1];
        }), prim->lines.end());
        prim->lines.update();

        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto &ind = prim->tris[i];
            repair(ind[0]);
            repair(ind[1]);
            repair(ind[2]);
        }
        prim->tris->erase(std::remove_if(prim->tris.begin(), prim->tris.end(), [&] (auto const &ind) {
            return ind[0] == ind[1] || ind[0] == ind[2] || ind[1] == ind[2];
        }), prim->tris.end());

        for (size_t i = 0; i < prim->quads.size(); i++) {
            auto &ind = prim->quads[i];
            repair(ind[0]);
            repair(ind[1]);
            repair(ind[2]);
            repair(ind[3]);
        }
        std::vector<uint8_t> ridquad(prim->quads.size());
        auto ridquadit = ridquad.begin();
        for (auto ind: prim->quads) {
            auto *bit = std::addressof(ind[0]);
            auto *eit = bit + 4;
            auto *mit = std::unique(bit, eit);
            auto len = mit - bit;
            //if (len != 4) printf("%d\n", len);
            if (len == 3)
                prim->tris.emplace_back(ind[0], ind[1], ind[2]);
            *ridquadit++ = (len <= 3);
        }
        prim->tris.update();
        prim->quads->erase(std::remove_if(prim->quads.begin(), prim->quads.end(), [&] (auto const &ind) {
            return ridquad[std::addressof(ind) - prim->quads.data()];
        }), prim->quads.end());
        prim->quads.update();

        for (size_t i = 0; i < prim->loops.size(); i++) {
            auto &ind = prim->loops[i];
            repair(ind);
        }
        for (auto &[base, len]: prim->polys) {
            auto bit = prim->loops.begin() + base;
            auto eit = prim->loops.begin() + (base + len);
            auto mit = std::unique(bit, eit);
            std::fill(mit, eit, 0); // not used anyway... prune later
            len = mit - bit;
        }
        prim->polys->erase(std::remove_if(prim->polys.begin(), prim->polys.end(), [&] (auto const &ply) {
            return ply[1] <= 2;
        }), prim->polys.end());
        prim->polys.update();

        prim->resize(nrevamp);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimWeld, {
    {
    {"PrimitiveObject", "prim", "", PrimarySocket},
    {"string", "tagAttr", "weld"},
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
