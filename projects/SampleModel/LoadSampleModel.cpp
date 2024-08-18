#include <zeno/zeno.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno_SampleModel_config.h>
#include <filesystem>
#include <sstream>
#include <cctype>
#ifdef __linux__
#include <unistd.h>
#include <stdio.h>
#endif

namespace zeno {
namespace {

std::string modelsDir() {
#ifdef __linux__
    char path[1024];
    getcwd(path, sizeof(path));
    auto cur_path = std::string(path);
    return cur_path + "/models";
#else
    auto cur_path = std::string(_pgmptr);
    cur_path = cur_path.substr(0, cur_path.find_last_of("\\"));
    return cur_path + "\\models";
#endif
}

struct LoadSampleModel : INode {
    virtual void apply() override {
        auto name = get_input2<std::string>("name");
        //auto dir = getAssetDir(MODELS_DIR, name + ".obj");
        auto dir = getAssetDir(modelsDir(), name + ".obj");  //temp fix for zeno-727 release
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
        {"enum cube monkey sphere humannose Pig_Head temple star", "name", "cube"},
        {gParamType_Bool, "triangulate", "1"},
        {gParamType_Bool, "decodeUVs", "1"},
    },
    {{gParamType_Primitive, "prim"}},
    {},
    {"primitive"},
});

struct LoadStringPrim : INode {
    inline static thread_local std::shared_ptr<PrimitiveObject> cache[128];

    virtual void apply() override {
        std::string str;
        if (has_input2<NumericObject>("str")) {
            std::visit([&] (auto const &val) {
                str = zeno::to_string(val);
            }, get_input<NumericObject>("str")->value);
        } else {
            str = get_input2<std::string>("str");
        }
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
                //auto path = getAssetDir(MODELS_DIR, ss.str());
                auto path = getAssetDir(modelsDir(), ss.str());  //temp fix for zeno-727 release
                if (!std::filesystem::exists(path)) {
                    zeno::log_warn("LoadStringPrim got ASCII char not printable: {}", c);
                    continue;
                }
                prim = std::static_pointer_cast<PrimitiveObject>(getThisGraph()->callTempNode("ReadObjPrim", {
                    {"triangulate:", std::make_shared<NumericObject>(int(0))},
                    {"decodeUVs:", std::make_shared<NumericObject>(int(0))},
                    {"path", objectFromLiterial(path)},
                }).at("prim"));
                cache[c] = std::make_shared<PrimitiveObject>(*prim);
            }

            primTranslate(prim.get(), vec3f(stride * i, 0, 0));
            primsRaw.push_back(prim.get());
            prims.push_back(std::move(prim));
        }

        auto retPrim = primsRaw.size() == 1 ? prims[0] : primMerge(primsRaw, {});
        if (get_input2<bool>("triangulate")) {
            primTriangulate(retPrim.get());
        }
        set_output("prim", std::move(retPrim));
    }
};

ZENO_DEFNODE(LoadStringPrim)({
    {
        {gParamType_String, "str", "Zello World!"},
        {gParamType_Bool, "triangulate", "1"},
    },
    {{gParamType_Primitive, "prim"}},
    {},
    {"primitive"},
});

}
}
