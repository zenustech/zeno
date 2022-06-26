#include <zeno/zeno.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno_FBX_config.h>

namespace zeno {
namespace {

struct LoadSampleModel : INode {
    virtual void apply() override {
        auto name = get_input2<std::string>("name");
        set_output("prim", getThisGraph()->callTempNode("ReadObjPrim", {
            {"triangulate", objectFromLiterial(get_input2<bool>("triangulate"))},
            {"decodeUVs", objectFromLiterial(get_input2<bool>("decodeUVs"))},
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

}
}
