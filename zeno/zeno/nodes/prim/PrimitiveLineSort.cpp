#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/logger.h>
#include <unordered_map>
#include <cassert>

namespace zeno {

void primLineSort(PrimitiveObject *prim) {
    std::vector<int> visited;
    {
        std::unordered_multimap<int, int> v2l;
        for (int i = 0; i < prim->lines.size(); i++) {
            auto line = prim->lines[i];
            //log_debug("line {} {}", line[0], line[1]);
            v2l.emplace(line[1], i);
        }

        int nsorted = 0;
        visited.resize(prim->verts.size(), -1);

        auto visit = [&] (auto &visit, int vert) -> void {
            if (visited[vert] != -1)
                return;
            visited[vert] = -2;
            auto [it0, it1] = v2l.equal_range(vert);
            for (auto it = it0; it != it1; ++it) {
                auto line = prim->lines[it->second];
                //assert(line[1] == vert);
                auto next = line[0];
                visit(visit, next);
            }
            visited[vert] = nsorted++;
        };

        for (int i = 0; i < prim->lines.size(); i++) {
            auto line = prim->lines[i];
            visit(visit, line[1]);
        }

        log_debug("sorted {} of {}", nsorted, visited.size());
        if (nsorted != visited.size()) {
            for (int i = 0; i < visited.size(); i++) {
                if (visited[i] == -1) {
                    visited[i] = nsorted++;
                }
            }
            assert(nsorted == visited.size());
        }

        //for (int i = 0; i < visited.size(); i++) {
            //log_debug("{} -> {} = {} {} {}", i, visited[i], prim->verts[i][0], prim->verts[i][1], prim->verts[i][2]);
        //}
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
            /*if constexpr (std::is_same_v<vec3f, std::decay_t<decltype(arr[0])>>) {
                for (int i = 0; i < newArr.size(); i++) {
                    log_info("{} = {} {} {}", i, newArr[i][0], newArr[i][1], newArr[i][2]);
                }
            }*/
            std::swap(arr, newArr);
        };
        revampvec(prim->verts.values);
        prim->verts.foreach_attr([&] (auto const &key, auto &attr) {
            revampvec(attr);
        });

        //for (int i = 0; i < prim->verts.size(); i++) {
            //log_info("{} = {} {} {}", i, prim->verts[i][0], prim->verts[i][1], prim->verts[i][2]);
        //}
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
