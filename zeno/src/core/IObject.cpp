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

ZENO_API UserData &IObject::userData() const {
    if (!m_userData.has_value())
        m_userData.emplace<UserData>();
    return std::any_cast<UserData &>(m_userData);
}

ZENO_API void IObject::accept(IObjectVisitor *visitor) {
    visitor->visit(this);
}

#define _ZENO_PER_XMACRO(TypeName, ...) \
ZENO_API void IObjectVisitor::visit(TypeName *object) { \
    visit(reinterpret_cast<IObject *>(object)); \
}
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO

ZENO_API void IObjectVisitor::visit(IObject *object) {
    /* do nothing */
}

ZENO_API IObjectVisitor::~IObjectVisitor() = default;

}
