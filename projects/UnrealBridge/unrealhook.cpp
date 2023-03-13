#include "zeno/core/IObject.h"
#include "unrealhook.h"
#include "unrealregistry.h"
#include "zeno/PrimitiveObject.h"
#include <algorithm>

namespace unreal {
    const std::string gHeightFieldLabel = "HeightField";
}

void zeno::UnrealHook::fetchViewObject(const std::string const &inObjKey, const std::shared_ptr<IObject>& inData) {
    const auto* data = dynamic_cast<zeno::PrimitiveObject*>(inData.get());
    if (data == nullptr) return;

    size_t labelIdx = inObjKey.find(':');
    const std::string& nodeName = labelIdx >= inObjKey.size() ? inObjKey : std::string { inObjKey.substr(0, labelIdx) };

    auto hasLabel = [&nodeName] (const std::string& label) {
        auto it = std::search(
            nodeName.begin(), nodeName.end(),
            unreal::gHeightFieldLabel.begin(), unreal::gHeightFieldLabel.end()
        );
        return it != nodeName.end();
    };

    if (hasLabel(unreal::gHeightFieldLabel) && data->verts.has_attr("height")) {
        auto& heightAttrs = data->verts.attr<float>("height");
        UnrealHeightFieldSubject subject {
            inObjKey,
            static_cast<int64_t>(data->verts.size()),
            heightAttrs,
        };
        UnrealSubjectRegistry::getStatic().put(nodeName, subject);
        UnrealSubjectRegistry::getStatic().markDirty(true);
    }
}

