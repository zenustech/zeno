#pragma once
#include <zeno/core/IObject.h>

namespace zeno
{
    // reference: NumericObject.h by archibate
    template <class ValueT>
    struct ZsObject : IObjectClone<ZsObject<ValueT>>
    {
        ValueT value;

        ZsObject() = default;
        ZsObject(ValueT const &value) : value(value) {}

        ZsObject &get()
        {
            return value;
        }

        ZsObject const &get() const
        {
            return value;
        }

        template <class T>
        T get() const
        {
            return std::visit(
                [](auto const &val) -> T
                {
                    using V = std::decay_t<decltype(val)>;
                    if constexpr (!std::is_constructible_v<T, V>)
                    {
                        throw makeError<TypeError>(typeid(T), typeid(V), "ZsObject return::get<T>");
                    }
                    else
                    {
                        return T(val);
                    }
                },
                value);
        }

        template <class T>
        bool is() const
        {
            return std::holds_alternative<T>(value);
        }

        template <class T>
        void set(T const &x)
        {
            value = x;
        }
    };
}
