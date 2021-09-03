#include <zeno/core/IObject.h>

namespace zeno {

ZENO_API IObject::IObject() = default;
ZENO_API IObject::~IObject() = default;

ZENO_API std::shared_ptr<IObject> IObject::clone() const {
    return nullptr;
}

ZENO_API bool IObject::assign(IObject *other) {
    return false;
}

ZENO_API bool IObject::movefrom(IObject *other) {
    return false;
}

}
