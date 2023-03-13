#include "model/subject.h"
#include "unrealregistry.h"
#include "zeno/core/INode.h"
#include "zeno/core/defNode.h"
#include "zeno/logger.h"
#include "zeno/types/PrimitiveObject.h"

namespace zeno {

struct IUnrealDataStreamNode : INode {
    virtual EZenoSubjectType type() = 0;

    void apply() override {
        auto subjectName = get_input2<std::string>("subject");
        EZenoSubjectApplyResult result = ZenoSubjectRegistry::getStatic().apply(subjectName, type(), *this);
        if (result != EZenoSubjectApplyResult::Success) {
            log_error("Failed to load unreal subject (result {}).", static_cast<int8_t>(result));
        }
    }
};

struct UnrealHeightField : IUnrealDataStreamNode {

    EZenoSubjectType type() override { return EZenoSubjectType::HeightField; }

};

ZENO_DEFNODE(UnrealHeightField)({
    {
        {"string", "subject", ""},
    },
    {"prim"},
    {},
    {"unreal"},
});

}
