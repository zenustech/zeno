#include <zeno/core/Graph.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <zeno/funcs/LiterialConverter.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/extra/GraphException.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/zeno.h>
#include <stack>

namespace zeno {

using namespace rapidjson;

template <class T, bool HasVec = true>
static T generic_get(Value const &x) {
    auto cast = [&] {
        if constexpr (std::is_same_v<T, zany>) {
            return [&] (auto &&x) -> zany {
                return objectFromLiterial(std::forward<decltype(x)>(x));
            };
        } else {
            return [&] (auto &&x) { return x; };
        };
    }();
    if (x.IsString()) {
        return cast((std::string)x.GetString());
    } else if (x.IsInt()) {
        return cast(x.GetInt());
    } else if (x.IsDouble()) {
        return cast((float)x.GetDouble());
    } else if (x.IsBool()) {
        return cast(x.GetBool());
    } else {
        if (x.IsObject()) {
            return parseObjectFromUi(x.GetObject());
        }
        if constexpr (HasVec) {
            if (x.IsArray()) {
                auto a = x.GetArray();
                if (a.Size() == 2) {
                    if (a[0].IsInt()) {
                        return cast(vec2i(a[0].GetInt(), a[1].GetInt()));
                    } else if (a[0].IsDouble()) {
                        return cast(vec2f(a[0].GetDouble(), a[1].GetDouble()));
                    }
                } else if (a.Size() == 3) {
                    if (a[0].IsInt()) {
                        return cast(vec3i(a[0].GetInt(), a[1].GetInt(), a[2].GetInt()));
                    } else if (a[0].IsDouble()) {
                        return cast(vec3f(a[0].GetDouble(), a[1].GetDouble(), a[2].GetDouble()));
                    }
                } else if (a.Size() == 4) {
                    if (a[0].IsInt()) {
                        return cast(vec4i(a[0].GetInt(), a[1].GetInt(), a[2].GetInt(), a[3].GetInt()));
                    } else if (a[0].IsDouble()) {
                        return cast(vec4f(a[0].GetDouble(), a[1].GetDouble(), a[2].GetDouble(), a[3].GetDouble()));
                    }
                }
            }
        }
        log_warn("unknown type encountered in generic_get");
        return cast(0);
    }
}

ZENO_API void Graph::loadGraph(const char *json) {
    //DEPRECATED
}

}
