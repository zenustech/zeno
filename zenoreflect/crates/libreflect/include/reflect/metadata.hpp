#pragma once

#include <cstdint>
#include <cstddef>
#include "reflect/macro.hpp"
#include "reflect/container/unique_ptr"

namespace zeno
{
namespace reflect
{
    class LIBREFLECT_API IMetadataValue {
    public:
        static UniquePtr<IMetadataValue> create_string(const char* str);
        static UniquePtr<IMetadataValue> create_list();

        virtual ~IMetadataValue();

        virtual bool is_string() const;
        virtual const char* as_string() const;

        virtual bool is_list() const;
        virtual size_t list_length() const;
        virtual const IMetadataValue* list_get_item(size_t index) const;
        virtual void list_add_item(UniquePtr<IMetadataValue>&& value);
    };

    class LIBREFLECT_API IRawMetadata {
    public:
        static UniquePtr<IRawMetadata> create();

        virtual ~IRawMetadata();

        virtual const IMetadataValue* get_value(const char* key) const = 0;

        virtual void set_value(const char* key, UniquePtr<IMetadataValue>&& value) = 0;
    };
}
}
