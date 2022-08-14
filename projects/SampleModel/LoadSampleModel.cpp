#include <zeno/zeno.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno_SampleModel_config.h>
#include <filesystem>
#include <sstream>
#include <cctype>

namespace zeno {
namespace {

struct LoadSampleModel : INode {
    virtual void apply() override {
        auto name = get_input2<std::string>("name");
        auto dir = getAssetDir(MODELS_DIR, name + ".obj");
        log_info("found sample model path [{}]", dir);
        set_output("prim", getThisGraph()->callTempNode("ReadObjPrim", {
            {"triangulate:", objectFromLiterial(get_input2<bool>("triangulate"))},
            {"decodeUVs:", objectFromLiterial(get_input2<bool>("decodeUVs"))},
            {"path", objectFromLiterial(dir)},
        }).at("prim"));
    }
};

ZENO_DEFNODE(LoadSampleModel)({
    {
        {"bool", "triangulate", "1"},
        {"bool", "decodeUVs", "1"},
        {"enum cube monkey sphere humannose Pig_Head temple star", "name", "cube"},
    },
    {"prim"},
    {},
    {"primitive"},
});

struct LoadStringPrim : INode {
    inline static std::shared_ptr<PrimitiveObject> cache[128];

    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        const float stride = 0.6f;

        std::vector<std::shared_ptr<PrimitiveObject>> prims;
        std::vector<PrimitiveObject *> primsRaw;
        for (int i = 0; i < str.size(); i++) {
            int c = str[i];
            if (std::isblank(c)) continue;

            std::shared_ptr<PrimitiveObject> prim;
            if (0 <= c && c < 128 && cache[c] && 0) {
                prim = std::make_shared<PrimitiveObject>(*cache[c]);
            } else {
                std::ostringstream ss;
                ss << "ascii/" << std::setw(3) << std::setfill('0') << c << ".obj";
                //printf("asdasd %s\n", ss.str().c_str());
                auto path = getAssetDir(MODELS_DIR, ss.str());
                if (!std::filesystem::exists(path)) {
                    zeno::log_warn("LoadStringPrim got ASCII char not printable: {}", c);
                    continue;
                }
                prim = std::static_pointer_cast<PrimitiveObject>(getThisGraph()->callTempNode("ReadObjPrim", {
                    {"triangulate:", objectFromLiterial(bool(0))},
                    {"decodeUVs:", objectFromLiterial(bool(0))},
                    {"path", objectFromLiterial(path)},
                }).at("prim"));
                cache[c] = std::make_shared<PrimitiveObject>(*prim);
            }

            primTranslate(prim.get(), vec3f(stride * i, 0, 0));
            primsRaw.push_back(prim.get());
            prims.push_back(std::move(prim));
        }

        auto retPrim = primsRaw.size() == 1 ? prims[0] : primMerge(primsRaw, {});
        if (get_input2<bool>("decodeUVs")) {
            primDecodeUVs(retPrim.get());
        }
        if (get_input2<bool>("triangulate")) {
            primTriangulate(retPrim.get());
        }
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
