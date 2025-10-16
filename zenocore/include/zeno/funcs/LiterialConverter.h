#pragma once

#include <zeno/utils/Error.h>
#include <zeno/core/IObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/safe_dynamic_cast.h>

namespace zeno {

template <class T>
inline bool objectIsLiterial(IObject* ptr) {
    if constexpr (std::is_base_of_v<IObject, T>) {
        return dynamic_cast<T *>(ptr);
    } else if constexpr (std::is_same_v<std::string, T>) {
        return dynamic_cast<StringObject *>(ptr);
    } else if constexpr (std::is_same_v<NumericValue, T>) {
        return dynamic_cast<NumericObject *>(ptr);
    } else {
        auto p = dynamic_cast<NumericObject *>(ptr);
        return p && std::visit([&] (auto const &val) -> bool {
            return std::is_constructible_v<T, std::decay_t<decltype(val)>>;
        }, p->value);
    }
}

template <class T>
inline bool objectIsRawLiterial(IObject* ptr) {
    if constexpr (std::is_base_of_v<IObject, T>) {
        return dynamic_cast<T *>(ptr);
    } else if constexpr (std::is_same_v<std::string, T>) {
        return dynamic_cast<StringObject *>(ptr);
    } else if constexpr (std::is_same_v<NumericValue, T>) {
        return dynamic_cast<NumericObject *>(ptr);
    } else {
        auto p = dynamic_cast<NumericObject *>(ptr);
        return p && std::visit([&] (auto const &val) -> bool {
            return std::is_same_v<T, std::decay_t<decltype(val)>>;
        }, p->value);
    }
}

template <class T>
inline auto objectToLiterial_old(std::shared_ptr<IObject> const& ptr, std::string const& msg = "objectToLiterial") {
    if constexpr (std::is_base_of_v<IObject, T>) {
        return safe_dynamic_cast<T>(ptr, msg);
    }
    else if constexpr (std::is_same_v<std::string, T>) {
        return safe_dynamic_cast<StringObject>(ptr.get(), msg)->get();
    }
    else if constexpr (std::is_same_v<NumericValue, T>) {
        return safe_dynamic_cast<NumericObject>(ptr.get(), msg)->get();
    }
    else if constexpr (std::is_same_v<glm::mat3, T>) {
        return std::get<glm::mat3>(safe_dynamic_cast<MatrixObject>(ptr.get(), msg)->m);
    }
    else if constexpr (std::is_same_v<glm::mat4, T>) {
        return std::get<glm::mat4>(safe_dynamic_cast<MatrixObject>(ptr.get(), msg)->m);
    }
    else {
        return std::visit([&](auto const& val) -> T {
            using T1 = std::decay_t<decltype(val)>;
            if constexpr (std::is_constructible_v<T, T1>) {
                return T(val);
            }
            else {
                throw makeError<TypeError>(typeid(T), typeid(T1), msg);
            }
            }, safe_dynamic_cast<NumericObject>(ptr.get(), msg)->get());
    }
}

template <class T>
inline auto objectToLiterial(const std::unique_ptr<IObject>& ptr, std::string const &msg = "objectToLiterial") {
    if constexpr (std::is_base_of_v<IObject, T>) {
        return safe_uniqueptr_cast<T>(ptr->clone());
    } else if constexpr (std::is_same_v<std::string, T>) {
        return safe_dynamic_cast<StringObject>(ptr.get(), msg)->get();
    } else if constexpr (std::is_same_v<NumericValue, T>) {
        return safe_dynamic_cast<NumericObject>(ptr.get(), msg)->get();
    } else if constexpr (std::is_same_v<glm::mat3, T>) {
        return std::get<glm::mat3>(safe_dynamic_cast<MatrixObject>(ptr.get(), msg)->m);
    } else if constexpr (std::is_same_v<glm::mat4, T>) {
        return std::get<glm::mat4>(safe_dynamic_cast<MatrixObject>(ptr.get(), msg)->m);
    }
    else {
        return std::visit([&] (auto const &val) -> T {
            using T1 = std::decay_t<decltype(val)>;
            if constexpr (std::is_constructible_v<T, T1>) {
                return T(val);
            } else {
                throw makeError<TypeError>(typeid(T), typeid(T1), msg);
            }
        }, safe_dynamic_cast<NumericObject>(ptr.get(), msg)->get());
    }
}

inline zany objectFromLiterial(std::string const &value) {
    return std::make_unique<StringObject>(value);
}

inline zany objectFromLiterial(NumericValue const &value) {
    return std::make_unique<NumericObject>(value);
}

inline zany objectFromLiterial(zany&& value) {
    return value->clone();
}

}
