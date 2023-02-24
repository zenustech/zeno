#include "zeno/core/IObject.h"
#include "unrealhook.h"


#ifdef ZENO_ENABLE_UNREALENGINE
    void zeno::UnrealHook::fetchViewObject(const std::string const &inObjKey, std::shared_ptr<IObject> inData) {
    }
#else // !ZENO_ENABLE_UNREALENGINE
    void zeno::UnrealHook::fetchViewObject(const std::string &inObjKey, std::shared_ptr<zeno::IObject> inData) {}
#endif // ZENO_ENABLE_UNREALENGINE

