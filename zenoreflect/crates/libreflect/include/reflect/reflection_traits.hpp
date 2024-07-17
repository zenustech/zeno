#pragma once

#include "type"

namespace zeno
{
namespace reflect
{
    class LIBREFLECT_API IReflectedObject {
    public:
        virtual ~IReflectedObject() = default;

        virtual TypeHandle type_info() {
            return TypeHandle::nulltype();
        }
    };

    template <typename T>
    class TEnableVirtualRefectionInfo : public IReflectedObject {
        virtual TypeHandle type_info() {
            return zeno::reflect::get_type<T>();
        }
    };
}
}
