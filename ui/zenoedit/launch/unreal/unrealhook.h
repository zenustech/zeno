#ifndef ZENO_UNREALHOOK_H
#define ZENO_UNREALHOOK_H

#include <string>
#include <memory>

namespace zeno {
    class IObject;

    struct UnrealHook {
        static void fetchViewObject(const std::string& inObjKey, std::shared_ptr<IObject> inData);
    };

}

#endif //ZENO_UNREALHOOK_H
