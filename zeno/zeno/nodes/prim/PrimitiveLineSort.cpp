#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/variantswitch.h>
#include <unordered_map>
#include <cassert>

namespace zeno {

void primLineSort(PrimitiveObject *prim) {
    std::vector<int> visited;
    {
        std::unordered_multimap<int, int> v2l;
        for (int i = 0; i < prim->lines.size(); i++) {
            auto line = prim->lines[i];
            v2l.emplace(line[0], i);
        }

        std::vector<int> tovisit;
        tovisit.reserve(prim->lines.size());
        for (int i = 0; i < prim->lines.size(); i++) {
            auto line = prim->lines[i];
            tovisit.push_back(line[0]);
        }

        int nsorted = 0;
        visited.resize(prim->verts.size(), -1);
        while (!tovisit.empty()) {
            int vert = tovisit.back();
            tovisit.pop_back();
            if (visited.at(vert) != -1)
                continue;
            visited[vert] = nsorted++;
            auto [it0, it1] = v2l.equal_range(vert);
            for (auto it = it0; it != it1; ++it) {
                auto line = prim->lines[it->second];
                auto next = line[1];
                tovisit.push_back(next);
            }
        }
        for (int i = 0; i < visited.size(); i++) {
            if (visited[i] == -1) {
                visited[i] = nsorted++;
            }
        }
        assert(nsorted == visited.size());
    }

    {
        auto revamp = [&] (int &x) { x = visited[x]; };
        for (auto &point: prim->points) {
            revamp(point);
        }
        for (auto &line: prim->lines) {
            revamp(line[0]);
            revamp(line[1]);
        }
        for (auto &tri: prim->tris) {
            revamp(tri[0]);
            revamp(tri[1]);
            revamp(tri[2]);
        }
        for (auto &quad: prim->quads) {
            revamp(quad[0]);
            revamp(quad[1]);
            revamp(quad[2]);
        }
        for (auto &loop: prim->loops) {
            revamp(loop);
        }
    }

    {
        auto revampvec = [&] (auto &arr) {
            std::vector<std::decay_t<decltype(arr[0])>> newArr(arr.size());
            for (int i = 0; i < arr.size(); i++) {
                int j = visited[i];
                newArr[j] = arr[i];
            }
            std::swap(arr, newArr);
        };
        revampvec(prim->verts.values);
        prim->foreach_attr([&] (auto const &key, auto &attr) {
            revampvec(attr);
        });
    }
}

struct PrimitiveLineSort : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        primLineSort(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveLineSort, {
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
