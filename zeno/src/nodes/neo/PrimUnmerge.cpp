#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/para/parallel_reduce.h>
#include <zeno/para/parallel_for.h>
#include <unordered_map>

namespace zeno {
namespace {

struct PrimUnmerge : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        auto method = get_input<StringObject>("method")->get();
        std::vector<std::shared_ptr<PrimitiveObject>> primList;

        if (prim->size() != 0) {
            if (method == "verts") {
                auto const &tagArr = prim->verts.attr<int>(tagAttr);
                int tagMax = -1;
                for (size_t i = 0; i < prim->verts.size(); i++) {
                    tagMax = std::max(tagMax, tagArr[i]);
                }
                tagMax++;

                primList.resize(tagMax);
                for (int tag = 0; tag < tagMax; tag++) {
                    primList[tag] = std::make_shared<PrimitiveObject>();
                }

                for (size_t i = 0; i < prim->verts.size(); i++) {
                    int tag = tagArr[i];
                    primList[tag]->verts.push_back(prim->verts[i]);
                }

            } else if (method == "faces") {
                auto const &tagArr = prim->tris.attr<int>(tagAttr);
                int tagMax = parallel_reduce_max(tagArr.begin(), tagArr.end());

                primList.resize(tagMax);
                for (int tag = 0; tag < tagMax; tag++) {
                    primList[tag] = std::make_shared<PrimitiveObject>();
                }

                std::vector<std::unordered_map<int, int>> vert_lut(tagMax);
                std::vector<std::vector<int>> vert_revamp(tagMax);

                auto mock = [&] (int tag, int &idx) {
                    auto it = vert_lut[tag].find(idx);
                    if (it == vert_lut[tag].end()) {
                        int new_idx = vert_revamp[tag].size();
                        vert_revamp[tag].push_back(idx);
                        vert_lut[tag].emplace(idx, new_idx);
                        idx = new_idx;
                    } else {
                        idx = it->second;
                    }
                };

                for (size_t i = 0; i < prim->lines.size(); i++) {
                    int tag = tagArr[i];
                    auto line = prim->lines[i];
                    mock(tag, line[0]);
                    mock(tag, line[1]);
                    primList[tag]->lines.push_back(line);
                }

                for (size_t i = 0; i < prim->tris.size(); i++) {
                    int tag = tagArr[i];
                    auto tri = prim->tris[i];
                    mock(tag, tri[0]);
                    mock(tag, tri[1]);
                    mock(tag, tri[2]);
                    primList[tag]->tris.push_back(tri);
                }

                for (size_t i = 0; i < prim->quads.size(); i++) {
                    int tag = tagArr[i];
                    auto quad = prim->quads[i];
                    mock(tag, quad[0]);
                    mock(tag, quad[1]);
                    mock(tag, quad[2]);
                    mock(tag, quad[3]);
                    primList[tag]->quads.push_back(quad);
                }

                for (size_t i = 0; i < prim->polys.size(); i++) {
                    int tag = tagArr[i];
                    auto poly = prim->polys[i];
                    int loopbegin = primList[tag]->loops.size();
                    for (int p = poly.first; p < poly.first + poly.second; p++) {
                        int loop = prim->loops[p];
                        mock(tag, loop);
                        primList[tag]->loops.push_back(loop);
                    }
                    primList[tag]->polys.push_back({loopbegin, poly.second});
                }

                parallel_for((int)0, tagMax, [&] (int tag) {
                    auto &revamp = vert_revamp[tag];
                    auto primOut = primList[tag].get();
                    primOut->verts.resize(revamp.size());
                    parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                        primOut->verts[i] = prim->verts[revamp[i]];
                    });
                    prim->verts.foreach_attr([&] (auto const &key, auto &arr) {
                        using T = std::decay_t<decltype(arr[0])>;
                        auto &outarr = primOut->verts.add_attr<T>(key);
                        parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                            outarr[i] = arr[revamp[i]];
                        });
                    });
                });
            }
        }

        auto listPrim = std::make_shared<ListObject>();
        for (auto &primPtr: primList) {
            listPrim->arr.push_back(std::move(primPtr));
        }
        set_output("listPrim", std::move(listPrim));
    }
};

ZENDEFNODE(PrimUnmerge, {
    {
        {"primitive", "prim"},
        {"string", "tagAttr", "tag"},
        {"enum verts faces", "method", "verts"},
    },
    {
        {"list", "listPrim"},
    },
    {
    },
    {"primitive"},
});

}
}
