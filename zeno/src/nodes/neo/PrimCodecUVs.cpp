#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/log.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace zeno {

// 'smart loop_uvs' to 'qianqiang loop.attr(uv)'
ZENO_API void primDecodeUVs(PrimitiveObject *prim) {
}

// 'smart loop_uvs' to 'veryqianqiang vert.attr(uv)'
ZENO_API void primLoopUVsToVerts(PrimitiveObject *prim) {
    if (prim->loops.size() && prim->has_attr("uvs")) {
        auto &loop_uvs = prim->loops.attr<int>("uvs");
        auto &vert_uv = prim->verts.add_attr<vec3f>("uv"); // todo: support vec2f in attr...
        /*attr_uv.resize(prim->loop_uvs.size());*/
        for (size_t i = 0; i < loop_uvs.size(); i++) {
            auto uv = prim->uvs[loop_uvs[i]];
            int vertid = prim->loops[i];
            vert_uv[vertid] = {uv[0], uv[1], 0};
            // uv may overlap and conflict at edges, but doesn't matter
            // this node is veryqianqiang after all, just to serve ZFX pw
        }
        prim->loops.erase_attr("uvs");
    }
}

namespace {

struct PrimDecodeUVs : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        primDecodeUVs(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimDecodeUVs)({
    {
        {"", "prim", "", zeno::Socket_ReadOnly}
    },
    {
        "prim",
    },
    {},
    {"deprecated"},
});

struct PrimLoopUVsToVerts : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        primLoopUVsToVerts(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimLoopUVsToVerts)({
    {
        {"", "prim", "", zeno::Socket_ReadOnly},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

struct PrimUVVertsToLoopsuv : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &vuv = prim->verts.attr<vec3f>("uv");
        if (prim->loops.size()) {
            auto &uvs = prim->loops.add_attr<int>("uvs");
            for (auto i = 0; i < prim->loops.size(); i++) {
                uvs[i] = prim->loops[i];
            }
            prim->uvs.resize(prim->verts.size());
            for (auto i = 0; i < prim->verts.size(); i++) {
                vec3f uv = vuv[i];
                prim->uvs[i] = {uv[0], uv[1]};
            }
        }
        else if (prim->tris.size()) {
            auto &uv0 = prim->tris.add_attr<vec3f>("uv0");
            auto &uv1 = prim->tris.add_attr<vec3f>("uv1");
            auto &uv2 = prim->tris.add_attr<vec3f>("uv2");
            for (auto i = 0; i < prim->tris.size(); i++) {
                uv0[i] = vuv[prim->tris[i][0]];
                uv1[i] = vuv[prim->tris[i][1]];
                uv2[i] = vuv[prim->tris[i][2]];
            }
        }
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimUVVertsToLoopsuv)({
    {
        {"", "prim", "", zeno::Socket_ReadOnly},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

struct PrimUVEdgeDuplicate : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto writeUVToVertex = get_input2<bool>("writeUVToVertex");
        bool isTris = prim->tris.size() > 0;
        if (isTris) {
            primPolygonate(prim.get(), true);
        }
        std::map<std::tuple<int, int>, int> mapping;
        std::vector<std::pair<int, int>> new_vertex_index;
        std::vector<int> new_loops;
        auto& uvs = prim->loops.attr<int>("uvs");
        for (auto i = 0; i < prim->loops.size(); i++) {
            int vi = prim->loops[i];
            int uvi = uvs[i];
            if (mapping.count({vi, uvi}) == 0) {
                int index = mapping.size();
                mapping[{vi, uvi}] = index;
                new_vertex_index.emplace_back(vi, uvi);
            }
            new_loops.push_back(mapping[{vi, uvi}]);
        }
        for (auto i = 0; i < prim->loops.size(); i++) {
            prim->loops[i] = new_loops[i];
        }

        AttrVector<vec3f> new_verts;
        new_verts.resize(mapping.size());
        for (auto i = 0; i < mapping.size(); i++) {
            int org = new_vertex_index[i].first;
            new_verts[i] = prim->verts[org];
        }
        prim->verts.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &attr = new_verts.add_attr<T>(key);
            for (auto i = 0; i < attr.size(); i++) {
                attr[i] = arr[new_vertex_index[i].first];
            }
        });
        std::swap(prim->verts, new_verts);
        if (writeUVToVertex) {
            auto &vert_uv = prim->verts.add_attr<vec3f>("uv");
            auto &loopsuv = prim->loops.attr<int>("uvs");
            for (auto i = 0; i < prim->loops.size(); i++) {
                auto uv = prim->uvs[loopsuv[i]];
                vert_uv[prim->loops[i]] = {uv[0], uv[1], 0};
            }
        }
        if (isTris) {
            primTriangulate(prim.get(), true, false);
        }

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimUVEdgeDuplicate)({
     {
         {"", "prim", "", zeno::Socket_ReadOnly},
         {"bool", "writeUVToVertex", "1"},
     },
     {
         "prim",
     },
     {},
     {"primitive"},
 });

struct PrimSplitVertexForSharedNormal : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        if (prim->loops.attr_is<vec3f>("nrm")) {
            std::vector<int> indexs;
            indexs.reserve(prim->loops.size());
            std::map<std::tuple<float, float, float>, int> mapping;
            {
                auto &nrm = prim->loops.attr<vec3f>("nrm");
                for (auto i = 0; i < prim->loops.size(); i++) {
                    std::tuple<float, float, float> n = {nrm[i][0], nrm[i][1], nrm[i][2]};
                    if (mapping.count(n) == 0) {
                        int count = mapping.size();
                        mapping[n] = count;
                    }
                    indexs.push_back(mapping[n]);
                }
                prim->loops.erase_attr("nrm");
            }
            std::map<int, vec3f> revert_mapping;
            for (auto [k, v]: mapping) {
                revert_mapping[v] = {std::get<0>(k), std::get<1>(k), std::get<2>(k)};
            }
            std::map<std::pair<int, int>, int> new_mapping;
            std::vector<int> new_indexs;
            for (auto i = 0; i < prim->loops.size(); i++) {
                std::pair<int, int> new_index = {prim->loops[i], indexs[i]};
                if (new_mapping.count(new_index) == 0) {
                    int count = new_mapping.size();
                    new_mapping[new_index] = count;
                }
                new_indexs.push_back(new_mapping[new_index]);
            }
            std::map<int, std::pair<int, int>> revert_new_mapping;
            for (auto [k, v]: new_mapping) {
                revert_new_mapping[v] = k;
            }
            AttrVector<vec3f> verts(new_mapping.size());
            auto &nrm = verts.add_attr<vec3f>("nrm");
            for (auto i = 0; i < verts.size(); i++) {
                verts[i] = prim->verts[revert_new_mapping[i].first];
                nrm[i] = revert_mapping[revert_new_mapping[i].second];
            }
            prim->verts.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                zeno::log_info("key: {}", key);
                auto &attr = verts.add_attr<T>(key);
                for (auto i = 0; i < attr.size(); i++) {
                    attr[i] = arr[revert_new_mapping[i].first];
                }
            });

            prim->verts = verts;
            for (auto i = 0; i < prim->loops.size(); i++) {
                prim->loops[i] = new_indexs[i];
            }
        }
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimSplitVertexForSharedNormal)({
    {
        "prim",
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});
}

}
