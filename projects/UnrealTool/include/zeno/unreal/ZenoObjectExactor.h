#pragma once

#include "ZenoUnrealTypes.h"
#include <functional>

namespace zeno::unreal {

struct ZenoObjectExactorManager {
    static ZenoObjectExactorManager& Get();

    void Register(EParamType type, std::function<std::shared_ptr<IObject>(const std::any&)> provider);

    std::shared_ptr<IObject> Exact(EParamType type, const std::any& data);

private:
    std::map<EParamType, std::function<std::shared_ptr<IObject>(const std::any&)>> dispatchMap;
};

}
