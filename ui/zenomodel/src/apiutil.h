#ifndef __API_UTIL_H__
#define __API_UTIL_H__

#include <QVariant>
#include "enum.h"
#include <zeno/core/IObject.h>

class ApiUtil
{
public:
    static ZVARIANT qVarToStdVar(const QVariant& qvar, QString typeDesc = "");
    static QVariant stdVarToQVar(const ZVARIANT& var);
};

namespace zeno {

template <class T, bool HasVec = true>
T generic_get_qvar(QVariant const &qvar, const QString& typeDesc) {
    auto cast = [&] {
        if constexpr (std::is_same_v<T, zany>) {
            return [&] (auto &&qvar) -> zany {
                return objectFromLiterial(std::forward<decltype(qvar)>(qvar));
            };
        } else {
            return [&] (auto &&qvar) { return qvar; };
        };
    }();

    QVariant::Type type = qvar.type();
    switch (type) {
    case QVariant::Int: {
        return cast(qvar.toInt());
    }
    case QVariant::Bool: {
        return cast(qvar.toBool());
    }
    case QMetaType::Float: {
        return cast(qvar.toFloat());
    }
    case QVariant::Double: {
        return cast(qvar.toFloat());
    }
    case QVariant::String: {
        auto val = qvar.toString().toStdString();
        return cast(val);
    }
    case QVariant::UserType:
        if constexpr (HasVec) {
            if (qvar.userType() == QMetaTypeId<UI_VECTYPE>::qt_metatype_id())
            {
                UI_VECTYPE qVec = qvar.value<UI_VECTYPE>();
                if (qVec.size() == 2)
                {
                    if (typeDesc == "vec2i")
                        return cast(vec2i(qVec[0], qVec[1]));
                    else if (typeDesc == "vec2f")
                        return cast(vec2f(qVec[0], qVec[1]));
                }
                else if (qVec.size() == 3)
                {
                    if (typeDesc == "vec3i")
                        return cast(vec3i(qVec[0], qVec[1], qVec[2]));
                    else if (typeDesc == "vec3f")
                        return cast(vec3f(qVec[0], qVec[1], qVec[2]));
                }
                else if (qVec.size() == 4)
                {
                    if (typeDesc == "vec4i")
                        return cast(vec4f(qVec[0], qVec[1], qVec[2], qVec[3]));
                    else if (typeDesc == "vec4f")
                        return cast(vec4f(qVec[0], qVec[1], qVec[2], qVec[3]));
                }
            }
        }
    default:
        break;
    }
    log_warn("unknown type encountered in generic_get");
    return cast(0);
}

}

#endif