#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/log.h>

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
        "prim",
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
        "prim",
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
         "prim",
         {"bool", "writeUVToVertex", "1"},
     },
     {
         "prim",
     },
     {},
     {"primitive"},
 });

}

}
