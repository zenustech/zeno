#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/Exception.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

template <class T>
bool objectIsLiterial(std::shared_ptr<IObject> const &ptr) {
    if constexpr (std::is_same_v<std::string, T>) {
        return dynamic_cast<StringObject *>(ptr.get());
    } else if constexpr (std::is_same_v<NumericValue, T>) {
        return dynamic_cast<NumericObject *>(ptr.get());
    } else {
        auto p = dynamic_cast<NumericObject *>(ptr.get());
        return p && std::visit([&] (auto const &val) -> bool {
            return std::is_constructible_v<T, std::decay_t<decltype(val)>>;
        }, p->value);
    }
}

template <class T>
T objectToLiterial(std::shared_ptr<IObject> const &ptr) {
    if constexpr (std::is_same_v<std::string, T>) {
        return safe_dynamic_cast<StringObject>(ptr.get())->get();
    } else if constexpr (std::is_same_v<NumericValue, T>) {
        return safe_dynamic_cast<NumericObject>(ptr.get())->get();
    } else {
        return std::visit([&] (auto const &val) -> T {
            if constexpr (std::is_constructible_v<T, std::decay_t<decltype(val)>>) {
                return T(val);
            } else {
                throw Exception((std::string)"invalid numeric cast to `" + typeid(T).name() + "`");
            }
        }, safe_dynamic_cast<NumericObject>(ptr.get())->get());
    }
}

inline std::shared_ptr<IObject> objectFromLiterial(std::string const &value) {
    return std::make_shared<StringObject>(value);
}

inline std::shared_ptr<IObject> objectFromLiterial(NumericValue const &value) {
    return std::make_shared<NumericObject>(value);
}

}
