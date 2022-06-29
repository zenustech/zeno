#include <zeno/zeno.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/utils/filesystem.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno_FBX_config.h>
#include <sstream>
#include <cctype>

namespace zeno {
namespace {

struct LoadSampleModel : INode {
    virtual void apply() override {
        auto name = get_input2<std::string>("name");
        set_output("prim", getThisGraph()->callTempNode("ReadObjPrim", {
            {"triangulate:", objectFromLiterial(get_input2<bool>("triangulate"))},
            {"decodeUVs:", objectFromLiterial(get_input2<bool>("decodeUVs"))},
            {"path", objectFromLiterial(getAssetDir(MODELS_DIR, name + ".obj"))},
        }).at("prim"));
    }
};

ZENO_DEFNODE(LoadSampleModel)({
    {
        {"bool", "triangulate", "1"},
        {"bool", "decodeUVs", "1"},
        {"enum cube monkey sphere humannose Pig_Head temple", "name", "cube"},
    },
    {"prim"},
    {},
    {"primitive"},
});

struct LoadStringPrim : INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");

        const float stride = 0.6f;

        std::vector<std::shared_ptr<PrimitiveObject>> prims;
        std::vector<PrimitiveObject *> primsRaw;
        for (int i = 0; i < str.size(); i++) {
            int c = str[i];
            if (std::isblank(c)) continue;

            std::ostringstream ss;
            ss << "ascii/" << std::setw(3) << std::setfill('0') << c << ".obj";
            //printf("asdasd %s\n", ss.str().c_str());
            auto path = getAssetDir(MODELS_DIR, ss.str());
            if (!fs::exists(path)) {
                zeno::log_warn("LoadStringPrim got ASCII char not printable: {}", c);
                continue;
            }
            auto prim = safe_dynamic_cast<PrimitiveObject>(getThisGraph()->callTempNode("ReadObjPrim", {
                {"triangulate:", objectFromLiterial(get_input2<bool>("triangulate"))},
                {"decodeUVs:", objectFromLiterial(get_input2<bool>("decodeUVs"))},
                {"path", objectFromLiterial(path)},
            }).at("prim"));

            primTranslate(prim.get(), vec3f(stride * i, 0, 0));
            primsRaw.push_back(prim.get());
            prims.push_back(std::move(prim));
        }

        auto retPrim = primMerge(primsRaw, {});
        set_output("prim", std::move(retPrim));
    }
};

ZENO_DEFNODE(LoadStringPrim)({
    {
        {"bool", "triangulate", "1"},
        {"bool", "decodeUVs", "1"},
        {"string", "str", "Zello World!"},
    },
    {"prim"},
    {},
    {"primitive"},
});

}
}
