#include <zeno/core/IObject.h>
#include <zeno/types/UserData.h>

namespace zeno {

ZENO_API IObject::IObject() = default;
ZENO_API IObject::IObject(IObject const &) = default;
ZENO_API IObject::IObject(IObject &&) = default;
ZENO_API IObject &IObject::operator=(IObject const &) = default;
ZENO_API IObject &IObject::operator=(IObject &&) = default;
ZENO_API IObject::~IObject() = default;

ZENO_API std::shared_ptr<IObject> IObject::clone() const {
    return nullptr;
}

ZENO_API std::shared_ptr<IObject> IObject::move_clone() {
    return nullptr;
}

ZENO_API bool IObject::assign(IObject *other) {
    return false;
}

ZENO_API bool IObject::move_assign(IObject *other) {
    return false;
}

ZENO_API UserData &IObject::userData() {
    return m_userData.access();
}

}
