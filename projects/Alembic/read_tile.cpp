// https://github.com/alembic/alembic/blob/master/lib/Alembic/AbcGeom/Tests/PolyMeshTest.cpp
// WHY THE FKING ALEMBIC OFFICIAL GIVES NO DOC BUT ONLY "TESTS" FOR ME TO LEARN THEIR FKING LIB
#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/fileio.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/GlobalState.h>
#include "rapidjson/document.h"


namespace zeno {
struct ReadTile : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto json = zeno::file_get_content(path);
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        const auto& root = doc["root"];
        const auto& children = root["children"];

        zeno::log_info("count {}", children.Size());

        for (auto i = 0; i < children.Size(); i++) {
            const auto &c = children[i];
            std::string uri = c["content"]["uri"].GetString();
            const auto &box = c["boundingVolume"]["box"];
            vec3f center = {
                    box[0].GetFloat(),
                    box[1].GetFloat(),
                    box[2].GetFloat(),
            };
            zeno::log_info("{}: {} {}", i, uri, center);
        }
    }
};

ZENDEFNODE(ReadTile, {
    {
        {"readpath", "path"},
    },
    {},
    {},
    {"alembic"},
});

struct LoadGLTFModel : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto json = zeno::file_get_content(path);
        rapidjson::Document doc;
        doc.Parse(json.c_str());
//        const auto& root = doc["root"];
//        const auto& children = root["children"];
//
//        zeno::log_info("count {}", children.Size());
//
//        for (auto i = 0; i < children.Size(); i++) {
//            const auto &c = children[i];
//            std::string uri = c["content"]["uri"].GetString();
//            const auto &box = c["boundingVolume"]["box"];
//            vec3f center = {
//                    box[0].GetFloat(),
//                    box[1].GetFloat(),
//                    box[2].GetFloat(),
//            };
//            zeno::log_info("{}: {} {}", i, uri, center);
//        }
    }
};

ZENDEFNODE(LoadGLTFModel, {
    {
        {"readpath", "path"},
    },
    {},
    {},
    {"alembic"},
});

} // namespace zeno
