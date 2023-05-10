#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/unreal/UnrealTool.h>

namespace zeno {

// Landscape input node
struct RemoteLandscapeInput : public INode {
    void apply() override {
        std::string SubjectName = get_input2<std::string>("SubjectName");
        std::optional<remote::HeightField> Data = remote::StaticRegistry.Get<remote::HeightField>(SubjectName, zeno::remote::StaticFlags.GetCurrentSession(), true);
        if (Data.has_value()) {
            std::shared_ptr<zeno::PrimitiveObject> Prim = remote::ConvertHeightDataToPrimitiveObject(Data.value());
            set_output2("prim", Prim);
        } else {
            zeno::log_error("Prim data not found.");
        }
    }
};

ZENO_DEFNODE(RemoteLandscapeInput)({
    {
        {"string", "SubjectName", "ReservedName_Landscape"},
    },
    {
        {"prim"},
    },
    {},
    {"Unreal"},
});

}
