#include "reflect/type.hpp"
#include "reflect/registry.hpp"
#include "reflect/utils/assert"
#include "type.hpp"

using namespace zeno::reflect;

TypeHandle::TypeHandle(TypeBase *type_info)
    : is_reflected_type(true)
{
    m_handle.type_info = type_info;
}

zeno::reflect::TypeHandle::TypeHandle(const RTTITypeInfo &rtti_info)
    : is_reflected_type(false)
{
    m_handle.rtti_hash = rtti_info.hash_code();
}

bool TypeHandle::operator==(const TypeHandle& other) const {
    return this->type_hash() == other.type_hash();
}

bool TypeHandle::operator!=(const TypeHandle& other) const {
    return !(other == *this);
}

TypeBase* TypeHandle::operator->() const
{
    return get_reflected_type_or_null();
}

size_t zeno::reflect::TypeHandle::type_hash() const
{
    if (is_reflected_type) {
        if (nullptr != m_handle.type_info) {
            return m_handle.type_info->type_hash();
        }
    } else {
        return m_handle.rtti_hash;
    }
    return 0;
}

TypeBase* TypeHandle::get_reflected_type_or_null() const
{
    if (is_reflected_type) {
        return m_handle.type_info;
    }
    TypeBase* reflected_type = ReflectionRegistry::get()->get(m_handle.rtti_hash);
    if (nullptr != reflected_type) {
        m_handle.type_info = reflected_type;
        is_reflected_type = true;
    }
    return reflected_type;
}


REFLECT_STATIC_CONSTEXPR TypeHandle::TypeHandle(const T_NullTypeArg&) {
    m_handle.rtti_hash = 0;
    is_reflected_type = false;
}

TypeBase::TypeBase(const ReflectedTypeInfo &type_info)
    : m_type_info(type_info)
{
}

bool TypeBase::operator==(const TypeBase &other) const
{
    return type_hash() == other.type_hash();
}

bool zeno::reflect::TypeBase::operator!=(const TypeBase &other) const
{
    return !(other == *this);
}

const ReflectedTypeInfo& zeno::reflect::TypeBase::get_info() const
{
    return m_type_info;
}

ArrayList<ITypeConstructor *> zeno::reflect::TypeBase::get_constructor(const ArrayList<RTTITypeInfo>& types) const
{
    const ArrayList<ITypeConstructor*>& available_ctors = get_constructors();
    ArrayList<ITypeConstructor*> suitable_ctors{available_ctors.size()};
    for (auto* ctor : available_ctors) {
        if (ctor->is_suitable_with_params(types)) {
            suitable_ctors.add_item(ctor);
        }
    }
    return suitable_ctors;
}

ITypeConstructor* zeno::reflect::TypeBase::get_constructor_or_null(const ArrayList<RTTITypeInfo>& params) const
{
    const ArrayList<ITypeConstructor*>& available_ctors = get_constructors();
    for (ITypeConstructor* ctor : available_ctors) {
        if (ctor->is_suitable_with_params(params)) {
            return ctor;
        }
    }
    return nullptr;
}

ITypeConstructor &zeno::reflect::TypeBase::get_constructor_checked(const ArrayList<RTTITypeInfo> &params) const
{
    ITypeConstructor* ctor = get_constructor_or_null(params);
    ZENO_CHECK(nullptr != ctor);
    return *ctor;
}

zeno::reflect::ITypeConstructor::~ITypeConstructor() = default;

zeno::reflect::ITypeConstructor::ITypeConstructor(TypeHandle in_type)
    : IBelongToParentType(in_type)
{
}

zeno::reflect::IHasParameter::~IHasParameter() = default;

bool zeno::reflect::IHasParameter::is_suitable_with_params(const ArrayList<RTTITypeInfo>& types) const
{
    const ArrayList<RTTITypeInfo>& signature_erased = get_params_dacayed();
    const ArrayList<RTTITypeInfo>& signature = get_params();
    if (types.size() < signature_erased.size()) {
        return false;
    }

    for (int i = 0; i < signature_erased.size(); ++i) {
        if (types[i] != signature_erased[i] && types[i] != signature[i]) {
            return false;
        }
    }

    return true;
}

bool zeno::reflect::IHasParameter::is_suitable_to_invoke(const ArrayList<Any> &params) const
{
    const ArrayList<RTTITypeInfo>& signature_erased = get_params_dacayed();
    if (params.size() < signature_erased.size()) {
        return false;
    }

    for (int i = 0; i < signature_erased.size(); ++i) {
        if (params[i].type() != signature_erased[i]) {
            return false;
        }
    }

    return true;
}

TypeHandle zeno::reflect::IBelongToParentType::get_parent_type() const
{
    return m_type;
}

zeno::reflect::IBelongToParentType::~IBelongToParentType()
{
}

zeno::reflect::IBelongToParentType::IBelongToParentType(TypeHandle in_type)
    : m_type(zeno::reflect::move(in_type))
{
}

zeno::reflect::IMemberFunction::~IMemberFunction()
{
}

zeno::reflect::IMemberFunction::IMemberFunction(TypeHandle in_type)
    : IBelongToParentType(in_type)
{
}

zeno::reflect::IHasName::~IHasName()
{
}

zeno::reflect::IMemberField::~IMemberField()
{
}

zeno::reflect::IMemberField::IMemberField(TypeHandle in_type)
    : IBelongToParentType(in_type)
{
}

zeno::reflect::IHasQualifier::~IHasQualifier()
{
}

bool zeno::reflect::IHasQualifier::is_static() const
{
    return false;
}

bool zeno::reflect::IHasQualifier::is_const() const
{
    return false;
}

bool zeno::reflect::IHasQualifier::is_volatile() const
{
    return false;
}

bool zeno::reflect::IHasQualifier::is_no_except() const
{
    return false;
}

bool zeno::reflect::IHasQualifier::is_mutable() const
{
    return false;
}

zeno::reflect::ICanHasMetadata::~ICanHasMetadata()
{
}

const IRawMetadata *zeno::reflect::ICanHasMetadata::get_metadata() const
{
    return nullptr;
}
