//
// Created by zh on 2024/9/18.
//
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/string.h>
#include <zeno/utils/bit_operations.h>
#include <zeno/types/PrimitiveUtils.h>
#include <boost/algorithm/string.hpp>

#include "Evaluate.h"
#include "zeno/utils/log.h"
#include "zeno/types/ListObject.h"
#include "zeno/funcs/PrimitiveUtils.h"

namespace zeno {
namespace zeno_nemo {
struct NemoObject : PrimitiveObject {
    std::unique_ptr<nemo::Evaluator> evaluator;
};

struct NemoEvaluator : INode {
    virtual void apply() override {
        auto path_config = get_input2<std::string>("Nemo Config");
        auto path_anim = get_input2<std::string>("Animation");
        auto nemo = std::make_shared<NemoObject>();
        nemo->evaluator = std::make_unique<nemo::Evaluator>(path_config, path_anim);
        nemo->userData().set2("path_config", path_config);
        nemo->userData().set2("path_anim", path_anim);
        set_output2("Evaluator", nemo);
    }
};

ZENDEFNODE(NemoEvaluator, {
    {
        { "readpath", "Nemo Config" },
        { "readpath", "Animation" },
    },
    {
        "Evaluator",
    },
    {},
    { "Nemo" },
});

static std::shared_ptr<ListObject> nemo_split_by_name(std::shared_ptr<PrimitiveObject> prim, bool add_when_none) {
    auto list = std::make_shared<ListObject>();
    if (prim->verts.size() == 0) {
        return list;
    }
    int faceset_count = prim->userData().get2<int>("faceset_count");
    if (add_when_none && faceset_count == 0) {
        auto name = prim->userData().get2<std::string>("abcpath_0");
        prim_set_faceset(prim.get(), name);
        faceset_count = 1;
    }
    std::map<int, std::vector<int>> faceset_map;
    for (auto f = 0; f < faceset_count; f++) {
        faceset_map[f] = {};
    }
    if (prim->polys.size()) {
        auto &faceset = prim->polys.add_attr<int>("faceset");
        for (auto j = 0; j < faceset.size(); j++) {
            auto f = faceset[j];
            faceset_map[f].push_back(j);
        }
        for (auto f = 0; f < faceset_count; f++) {
            auto name = prim->userData().get2<std::string>(zeno::format("faceset_{}", f));
            auto new_prim = std::dynamic_pointer_cast<PrimitiveObject>(prim->clone());
            new_prim->polys.resize(faceset_map[f].size());
            for (auto i = 0; i < faceset_map[f].size(); i++) {
                new_prim->polys[i] = prim->polys[faceset_map[f][i]];
            }
            new_prim->polys.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto &attr = prim->polys.attr<T>(key);
                for (auto i = 0; i < arr.size(); i++) {
                    arr[i] = attr[faceset_map[f][i]];
                }
            });
            for (auto j = 0; j < faceset_count; j++) {
                new_prim->userData().del(zeno::format("faceset_{}", j));
            }
            prim_set_faceset(new_prim.get(), name);
            list->arr.push_back(new_prim);
        }
    }
    return list;
}

struct NemoPlay : INode {
    virtual void apply() override {
        auto evaluator = get_input2<NemoObject>("Evaluator");
        if (!evaluator) {
            // TODO: return error();
        }
        float frame = getGlobalState()->frameid;
        if (has_input("frameid")) {
            frame = get_input2<float>("frameid");
        }
        evaluator->evaluator->evaluate(frame);
        bool skipHiddenPrim = get_input2<bool>("skipHiddenPrim");
        std::map<int, std::vector<int>> mapping_mesh_to_faceset;

        auto readFaceset = get_input2<bool>("readFaceset");
        if (readFaceset) {
            for (auto i = 0; i < evaluator->evaluator->facesets.size(); i++) {
                auto const &faceset = evaluator->evaluator->facesets[i];
                for (auto [mesh_id, _]: faceset.members) {
                    if (mapping_mesh_to_faceset.count(mesh_id) == 0) {
                        mapping_mesh_to_faceset[mesh_id] = {};
                    }
                    mapping_mesh_to_faceset[mesh_id].push_back(i);
                }
            }
        }

        auto splitByFaceset = get_input2<bool>("splitByFaceset");
        auto killDeadVerts = get_input2<bool>("killDeadVerts");
        auto triangulate = get_input2<bool>("triangulate");

        auto prims = std::make_shared<zeno::ListObject>();
        std::map<unsigned, std::string> meshes;
        for (unsigned mesh_id = 0; mesh_id != evaluator->evaluator->meshes.size(); ++mesh_id) {
            unsigned plug_id = evaluator->evaluator->meshes[mesh_id];
            if (skipHiddenPrim && evaluator->evaluator->isVisible(plug_id) == 0) {
                continue;
            }

            std::string path = evaluator->evaluator->LUT_path.at(plug_id);
            boost::algorithm::replace_all(path, "|", "/");
            meshes[plug_id] = path;
            std::vector<glm::vec3> points = evaluator->evaluator->getPoints(plug_id);
            auto sub_prim = std::make_shared<zeno::PrimitiveObject>();
            sub_prim->verts.resize(points.size());
            for (auto i = 0; i < points.size(); i++) {
                sub_prim->verts[i] = bit_cast<vec3f>(points[i]);
            }
            auto [counts, connection] = evaluator->evaluator->getTopo(plug_id);
            sub_prim->loops.reserve(counts.size());
            for (auto i: connection) {
                sub_prim->loops.push_back(i);
            }
            auto offset = 0;
            sub_prim->polys.reserve(counts.size());
            for (auto i: counts) {
                sub_prim->polys.emplace_back(offset, i);
                offset += i;
            }
            auto [uValues, vValues, uvIds] = evaluator->evaluator->getDefaultUV(plug_id);
            if (uvIds.size() == connection.size()) {
                auto &uvs = sub_prim->loops.add_attr<int>("uvs");
                for (auto i = 0; i < uvs.size(); i++) {
                    uvs[i] = uvIds[i];
                }
                sub_prim->uvs.reserve(uValues.size());
                for (auto i = 0; i < uValues.size(); i++) {
                    sub_prim->uvs.emplace_back(uValues[i], vValues[i]);
                }
            }
            std::vector<std::string> faceSetNames;
            if (readFaceset) {
                auto &faceset_attr = sub_prim->polys.add_attr<int>("faceset");
                std::fill(faceset_attr.begin(), faceset_attr.end(), -1);
                if (mapping_mesh_to_faceset.count(plug_id)) {
                    for (auto fi: mapping_mesh_to_faceset[plug_id]) {
                        auto &faceset = evaluator->evaluator->facesets[fi];
                        auto cur_index = faceSetNames.size();
                        faceSetNames.push_back(faceset.name);
                        for (auto i: faceset.members[plug_id]) {
                            faceset_attr[i] = cur_index;
                        }
                    }
                }
                for (auto i = 0; i < faceSetNames.size(); i++) {
                    auto n = faceSetNames[i];
                    sub_prim->userData().set2(zeno::format("faceset_{}", i), n);
                }
                sub_prim->userData().set2("faceset_count", int(faceSetNames.size()));
            }
            prim_set_abcpath(sub_prim.get(), "/ABC"+path);
            sub_prim->userData().set2("vis", int(evaluator->evaluator->isVisible(plug_id)));
            if (splitByFaceset && faceSetNames.size() >= 2) {
                auto list = nemo_split_by_name(sub_prim, true);
                for (auto p: list->arr) {
                    auto np = std::dynamic_pointer_cast<PrimitiveObject>(p);
                    if (killDeadVerts) {
                        primKillDeadVerts(np.get());
                    }
                    prims->arr.emplace_back(p);
                }
            }
            else {
                prims->arr.emplace_back(sub_prim);
            }
        }

        if (triangulate) {
            for (auto &prim: prims->arr) {
                auto _prim = std::dynamic_pointer_cast<PrimitiveObject>(prim);
                zeno::primTriangulate(_prim.get());
            }
        }

        set_output2("prims", prims);
    }
};


ZENDEFNODE(NemoPlay, {
    {
        { "Evaluator" },
        { "frame" },
        { "bool", "skipHiddenPrim", "1" },
        { "bool", "readFaceset", "1" },
        { "bool", "splitByFaceset", "0"},
        { "bool", "triangulate", "0"},
        { "bool", "killDeadVerts", "1"},
    },
    {
        "prims",
    },
    {},
    { "Nemo" },
});
}
}
