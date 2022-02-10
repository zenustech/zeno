#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <zeno/types/LiterialConverter.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>

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
    } else if (x.IsFloat()) {
        return cast(x.GetFloat());
    } else if (x.IsBool()) {
        return cast(x.GetBool());
    } else {
        if constexpr (HasVec) {
            if (x.IsArray()) {
                auto a = x.GetArray();
                if (a.Size() == 2) {
                    if (a[0].IsInt()) {
                        return cast(vec2i(a[0].GetInt(), a[1].GetInt()));
                    } else if (a[0].IsFloat()) {
                        return cast(vec2f(a[0].GetFloat(), a[1].GetFloat()));
                    }
                } else if (a.Size() == 3) {
                    if (a[0].IsInt()) {
                        return cast(vec3i(a[0].GetInt(), a[1].GetInt(), a[2].GetInt()));
                    } else if (a[0].IsFloat()) {
                        return cast(vec3f(a[0].GetFloat(), a[1].GetFloat(), a[2].GetFloat()));
                    }
                } else if (a.Size() == 4) {
                    if (a[0].IsInt()) {
                        return cast(vec4i(a[0].GetInt(), a[1].GetInt(), a[2].GetInt(), a[4].GetInt()));
                    } else if (a[0].IsFloat()) {
                        return cast(vec4f(a[0].GetFloat(), a[1].GetFloat(), a[2].GetFloat(), a[4].GetFloat()));
                    }
                }
            }
        }
        log_warn("unknown type encountered in generic_get");
        return cast(0);
    }
}

ZENO_API void Graph::loadGraph(const char *json) {
    Document d;
    d.Parse(json);

    for (int i = 0; i < d.Size(); i++) {
        Value const &di = d[i];
        std::string cmd = di[0].GetString();
#ifdef ZENO_FAIL_SILENTLY
        try {
#endif
            if (0) {
            } else if (cmd == "addNode") {
                addNode(di[1].GetString(), di[2].GetString());
            } else if (cmd == "completeNode") {
                completeNode(di[1].GetString());
            } else if (cmd == "setNodeInput") {
                setNodeInput(di[1].GetString(), di[2].GetString(), generic_get<zany>(di[3]));
            } else if (cmd == "setNodeParam") {
                setNodeParam(di[1].GetString(), di[2].GetString(), generic_get<std::variant<int, float, std::string>, false>(di[3]));
            } else if (cmd == "setNodeOption") {
                setNodeOption(di[1].GetString(), di[2].GetString());
            } else if (cmd == "bindNodeInput") {
                bindNodeInput(di[1].GetString(), di[2].GetString(), di[3].GetString(), di[4].GetString());
            } else {
                log_warn("got unexpected command: {}", cmd);
            }
#ifdef ZENO_FAIL_SILENTLY
        } catch (BaseException const &e) {
            log_warn("exception executing command {} ({}): {}",
                    i, cmd.c_str(), e.what());
        }
#endif
    }
}

}
