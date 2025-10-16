
#include <zeno/core/IObject.h>
#include <zeno/types/UserData.h>

namespace zeno {

    IObject::IObject() {
        m_usrData = std::make_unique<UserData>();
    }

    IObject::~IObject() {
    }

    IObject::IObject(const IObject& rhs) {
        m_key = rhs.m_key;
        m_usrData = rhs.m_usrData->clone();
    }

    IObject& IObject::operator=(const IObject& rhs) {
        m_key = rhs.m_key;
        m_usrData = rhs.m_usrData->clone();
        return *this;
    }

    String IObject::key() const {
        return m_key;
    }

    void IObject::update_key(const String& key) {
        m_key = key;
    }

    IUserData* IObject::userData() {
        return m_usrData.get();
    }

    void IObject::Delete() {
    }

#if 0
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

ZENO_API std::shared_ptr<IObject> IObject::clone_by_key(std::string const& prefix) {
    return nullptr;
}

ZENO_API bool IObject::assign(IObject const *other) {
    return false;
}

ZENO_API bool IObject::move_assign(IObject *other) {
    return false;
}

ZENO_API std::string IObject::method_node(std::string const &op) {
    return {};
}

ZENO_API std::string IObject::key() {
    return "";
}

ZENO_API bool IObject::update_key(const std::string& key) {
    return false;
}

ZENO_API void IObject::set_parent(IObject* spParent) {
    m_parent = spParent;
}

ZENO_API IObject* IObject::get_parent() const {
    return m_parent;
}

ZENO_API std::vector<std::string> IObject::paths() const {
    return {};
}

ZENO_API UserData &IObject::userData() const {
    if (!m_userData.has_value())
        m_userData.emplace<UserData>();
    return std::any_cast<UserData &>(m_userData);
}
#endif
}
