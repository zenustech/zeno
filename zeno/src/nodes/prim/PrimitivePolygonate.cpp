#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/utils/log.h>

namespace zeno {

ZENO_API void primPolygonate(PrimitiveObject *prim, bool with_uv) {
    prim->loops.reserve(prim->loops.size() + prim->tris.size() * 3 +
                        prim->quads.size() * 4 + prim->lines.size() * 2 +
                        prim->points.size());
    prim->polys.reserve(prim->polys.size() + prim->tris.size() +
                        prim->quads.size() + prim->lines.size() +
                        prim->points.size());
    bool tri_has_mat = prim->tris.has_attr("matid");
    bool quad_has_mat = prim->quads.has_attr("matid");
    std::vector<int> matid;
    matid.resize(prim->polys.size() + prim->tris.size() +
                 prim->quads.size() + prim->lines.size() +
                 prim->points.size());
    matid.assign(matid.size(), -1);

    bool tri_has_faceset = prim->tris.has_attr("faceset");
    bool quad_has_faceset = prim->quads.has_attr("faceset");
    std::vector<int> faceset;
    faceset.resize(prim->polys.size() + prim->tris.size() +
                 prim->quads.size() + prim->lines.size() +
                 prim->points.size());
    faceset.assign(faceset.size(), -1);

    bool tri_has_abcpath = prim->tris.has_attr("abcpath");
    bool quad_has_abcpath = prim->quads.has_attr("abcpath");
    std::vector<int> abcpath;
    abcpath.resize(prim->polys.size() + prim->tris.size() +
                 prim->quads.size() + prim->lines.size() +
                 prim->points.size());
    abcpath.assign(abcpath.size(), -1);

    int old_loop_base = prim->loops.size();
    int polynum = prim->polys.size();
    if (prim->tris.size()) {
        int base = prim->loops.size();
        for (int i = 0; i < prim->tris.size(); i++) {
            auto const &ind = prim->tris[i];
            prim->loops.push_back(ind[0]);
            prim->loops.push_back(ind[1]);
            prim->loops.push_back(ind[2]);
            prim->polys.push_back({base + i * 3, 3});
            if(tri_has_mat)
                matid[polynum + i] = prim->tris.attr<int>("matid")[i];
            if (tri_has_faceset)
                faceset[polynum + i] = prim->tris.attr<int>("faceset")[i];
            if (tri_has_abcpath)
                abcpath[polynum + i] = prim->tris.attr<int>("abcpath")[i];
        }

        prim->tris.foreach_attr([&](auto const &key, auto const &arr) {
            if (key == "uv0" || key == "uv1" || key == "uv2") {
                return;
            }
            using T = std::decay_t<decltype(arr[0])>;
            auto &newarr = prim->polys.add_attr<T>(key);
            newarr.insert(newarr.end(), arr.begin(), arr.end());
        });
    }

    polynum = prim->polys.size();
    if (prim->quads.size()) {
        int base = prim->loops.size();
        for (int i = 0; i < prim->quads.size(); i++) {
            auto const &ind = prim->quads[i];
            prim->loops.push_back(ind[0]);
            prim->loops.push_back(ind[1]);
            prim->loops.push_back(ind[2]);
            prim->loops.push_back(ind[3]);
            prim->polys.push_back({base + i * 4, 4});
            if(quad_has_mat)
                matid[polynum + i] = prim->quads.attr<int>("matid")[i];
            if (quad_has_faceset)
                faceset[polynum + i] = prim->quads.attr<int>("faceset")[i];
            if (quad_has_abcpath)
                abcpath[polynum + i] = prim->quads.attr<int>("abcpath")[i];
        }

        prim->quads.foreach_attr([&](auto const &key, auto const &arr) {
            if (key == "uv0" || key == "uv1" || key == "uv2" || key == "uv3") {
                return;
            }
            using T = std::decay_t<decltype(arr[0])>;
            auto &newarr = prim->polys.add_attr<T>(key);
            newarr.insert(newarr.end(), arr.begin(), arr.end());
        });
    }

    if (prim->lines.size()) {
        int base = prim->loops.size();
        for (int i = 0; i < prim->lines.size(); i++) {
            auto const &ind = prim->lines[i];
            prim->loops.push_back(ind[0]);
            prim->loops.push_back(ind[1]);
            prim->polys.push_back({base + i * 2, 2});
        }

        prim->lines.foreach_attr([&](auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &newarr = prim->polys.add_attr<T>(key);
            newarr.insert(newarr.end(), arr.begin(), arr.end());
        });
    }

    if (prim->points.size()) {
        int base = prim->loops.size();
        for (int i = 0; i < prim->points.size(); i++) {
            auto ind = prim->points[i];
            prim->loops.push_back(ind);
            prim->polys.push_back({base + i, 1});
        }

        prim->points.foreach_attr([&](auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &newarr = prim->polys.add_attr<T>(key);
            newarr.insert(newarr.end(), arr.begin(), arr.end());
        });
    }

    prim->loops.update();
    prim->polys.update();

    if (!(!prim->tris.has_attr("uv0") || !prim->tris.has_attr("uv1") ||
          !prim->tris.has_attr("uv2") || !with_uv)) {
        auto old_uvs_base = prim->uvs.size();
        prim->loops.add_attr<int>("uvs");
        auto &uv0 = prim->tris.attr<vec3f>("uv0");
        auto &uv1 = prim->tris.attr<vec3f>("uv1");
        auto &uv2 = prim->tris.attr<vec3f>("uv2");
        for (int i = 0; i < prim->tris.size(); i++) {
            prim->loops.attr<int>("uvs")[old_loop_base + i * 3 + 0] = old_uvs_base + i * 3 + 0;
            prim->loops.attr<int>("uvs")[old_loop_base + i * 3 + 1] = old_uvs_base + i * 3 + 1;
            prim->loops.attr<int>("uvs")[old_loop_base + i * 3 + 2] = old_uvs_base + i * 3 + 2;
            prim->uvs.emplace_back(uv0[i][0], uv0[i][1]);
            prim->uvs.emplace_back(uv1[i][0], uv1[i][1]);
            prim->uvs.emplace_back(uv2[i][0], uv2[i][1]);
        }
        // remove duplicate uv index
        {
            std::map<std::tuple<float, float>, int> mapping;
            auto &loopsuv = prim->loops.attr<int>("uvs");
            for (auto i = 0; i < prim->loops.size(); i++) {
                vec2f uv = prim->uvs[loopsuv[i]];
                if (mapping.count({uv[0], uv[1]}) == false) {
                    mapping[{uv[0], uv[1]}] = loopsuv[i];
                }
                loopsuv[i] = mapping[{uv[0], uv[1]}];
            }
        }
    }
    prim->polys.add_attr<int>("matid");
    for(int i=0;i<matid.size();i++)
    {
        prim->polys.attr<int>("matid")[i] = matid[i];
    }
    prim->polys.add_attr<int>("faceset");
    for(int i=0;i<faceset.size();i++)
    {
        prim->polys.attr<int>("faceset")[i] = faceset[i];
    }

    prim->polys.add_attr<int>("abcpath");
    for(int i=0;i<abcpath.size();i++)
    {
        prim->polys.attr<int>("abcpath")[i] = abcpath[i];
    }

    prim->tris.clear();
    prim->quads.clear();
    prim->lines.clear();
    prim->points.clear();
}

namespace {

struct PrimitivePolygonate : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        primPolygonate(prim.get(), get_param<bool>("with_uv"));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitivePolygonate,
        { /* inputs: */ {
        {"primitive", "prim"},
        }, /* outputs: */ {
        {"primitive", "prim"},
        }, /* params: */ {
        {"bool", "with_uv", "1"},
        }, /* category: */ {
        "primitive",
        }});

}
}
