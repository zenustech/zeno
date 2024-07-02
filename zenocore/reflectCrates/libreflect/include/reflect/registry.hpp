#pragma once

#include <cstdint>
#include "reflect/macro.hpp"
#include "reflect/type.hpp"
#include "reflect/container/arraylist"
#include "reflect/container/string"

#define REFLECT_REGISTER_RTTI_TYPE_MANUAL(Ty) \
    namespace zeno { \
    namespace reflect { \
        template <> \
        struct _manual_register_rtti_type_internal<Ty> {}; \
    }}
    

namespace zeno
{
namespace reflect 
{
    template <typename T>
    struct _manual_register_rtti_type_internal {};

    struct internal_utils {
        static LIBREFLECT_API int32_t allocate_new_id();
    };

    /**
     * This is a pimpl wrapper for a std::map for ABI compatibility
    */
    class LIBREFLECT_API ReflectTypeMap {
    private:
        using KeyType = size_t;
        using ValueType = TypeBase*;

        void* m_opaque_data = nullptr;
    public:
        ReflectTypeMap();
        ~ReflectTypeMap();

        bool add(ValueType val);
        size_t size() const;
        ValueType get(KeyType hash);
        ArrayList<ValueType> all() const;
        ValueType find_by_canonical_name(const StringView& in_view);
    };

    class LIBREFLECT_API ReflectionRegistry final {
    private:
        ReflectionRegistry() = default;

        ReflectTypeMap m_typed_map;
    public:
        static ReflectionRegistry& get();

        ReflectTypeMap* operator->();
    };
}
}

