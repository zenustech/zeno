//
// Created by zh on 2024/9/18.
//
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/string.h>
#include <zeno/utils/bit_operations.h>
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

struct NemoPlay : INode {
    virtual void apply() override {
        auto evaluator = get_input2<NemoObject>("Evaluator");
        if (!evaluator) {
            // TODO: return error();
        }
        float frame = getGlobalState()->frameid;
        if (has_input("frameid")) {
            frame = get_input<NumericObject>("frameid")->get<float>();
        }
        evaluator->evaluator->evaluate(frame);

        auto prims = std::make_shared<zeno::ListObject>();
        std::map<unsigned, std::string> meshes;
        for (unsigned mesh_id = 0; mesh_id != evaluator->evaluator->meshes.size(); ++mesh_id) {
            unsigned plug_id = evaluator->evaluator->meshes[mesh_id];
            std::string path = evaluator->evaluator->LUT_path.at(plug_id);
            boost::algorithm::replace_all(path, "|", "/");
            meshes[plug_id] = path;
            zeno::log_info("path: {}", path);
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
            prim_set_abcpath(sub_prim.get(), "/ABC"+path);
            prims->arr.emplace_back(sub_prim);
        }

        set_output2("prims", prims);
    }
};


ZENDEFNODE(NemoPlay, {
    {
        { "Evaluator" },
        { "frame" },
    },
    {
        "prims",
    },
    {},
    { "Nemo" },
});
}
}
