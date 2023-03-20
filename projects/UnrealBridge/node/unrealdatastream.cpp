#include "model/subject.h"
#include "unrealregistry.h"
#include "ubipcclient.h"
#include "zeno/core/INode.h"
#include "zeno/core/defNode.h"
#include "zeno/logger.h"
#include "zeno/types/PrimitiveObject.h"

namespace zeno {

struct IUnrealDataStreamNode : public INode {
    virtual EZenoSubjectType type() = 0;

    void apply() override = 0;
};

struct FetchUnrealHeightField : public INode {

//    EZenoSubjectType type()  { return EZenoSubjectType::HeightField; }

    void apply() override {
        auto subjectName = get_input2<std::string>("subject");
        std::optional<UnrealZenoHeightFieldSubject> subject = ::UnrealBridge::IPCClient::fetchSubject<UnrealZenoHeightFieldSubject>(subjectName);
        if (subject.has_value()) {
            UnrealZenoHeightFieldSubject* heightFieldSubject = &subject.value();
            if (heightFieldSubject) {
                std::shared_ptr<PrimitiveObject> prim = std::make_shared<PrimitiveObject>();
                prim->verts.resize(heightFieldSubject->heights.size());
                prim->verts.add_attr<float>("height");
                auto& heights = prim->verts.attr<float>("height");
                for (size_t i = 0; i < heightFieldSubject->heights.size(); ++i) {
                    heights[i] = heightFieldSubject->heights[i];
                }

                set_output2("prim", prim);
            } else {
                log_error("Subject '%s' isn't a height field", subjectName);
            }
        }
        else {
            log_error("Could not found subject with name '%s'", subjectName);
        }
    }
};

ZENO_DEFNODE(FetchUnrealHeightField)({
    {
        {"string", "subject", ""},
    },
    {"prim"},
    {},
    {"unreal"},
});

struct PushHeightFieldToUnreal : public INode {
    void apply() override {
        auto subjectName = get_input2<std::string>("subject");
        if (subjectName.empty()) {
            log_error("Subject name should not be empty.");
            return;
        }
        auto prim = get_input2<PrimitiveObject>("prim");
        if (!prim) {
            log_error("Primitive should not be empty.");
            return;
        }
        if (!prim->has_attr("height")) {
            log_error("Primitive input should have height attribute.");
            return;
        }
        auto& heightAttrs = prim->verts.attr<float>("height");
        UnrealHeightFieldSubject subject {
            subjectName,
            static_cast<int64_t>(prim->verts.size()),
            heightAttrs,
        };
        ::UnrealBridge::IPCClient::sendSubject<UnrealHeightFieldSubject>(subjectName, subject);
        set_output2("subject", subjectName);
        set_output2("prim", prim);
    }
};
ZENO_DEFNODE(PushHeightFieldToUnreal)({
    {
        {"prim"},
        {"string", "subject", "DefaultSubject"},
    },
    {
        { "prim" },
        {"string", "subject", "DefaultSubject"},
    },
    {},
    {"unreal"},
});

}
