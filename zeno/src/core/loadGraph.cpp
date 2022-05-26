#include <zeno/core/Graph.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <zeno/funcs/LiterialConverter.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/extra/GraphException.h>
#include <zeno/utils/Translator.h>
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
    } else if (x.IsDouble()) {
        return cast((float)x.GetDouble());
    } else if (x.IsBool()) {
        return cast(x.GetBool());
    } else {
        if constexpr (std::is_same_v<T, zany>) {
            if (x.IsObject()) {
                return parseObjectFromUi(x.GetObject());
            }
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
                        return cast(vec4i(a[0].GetInt(), a[1].GetInt(), a[2].GetInt(), a[4].GetInt()));
                    } else if (a[0].IsDouble()) {
                        return cast(vec4f(a[0].GetDouble(), a[1].GetDouble(), a[2].GetDouble(), a[4].GetDouble()));
                    }
                }
            }
        }
        log_warn("unknown type encountered in generic_get");
        return cast(0);
    }
}

ZENO_API void Graph::loadGraph(const char *json) {
    GraphException::catched([&] {
        Document d;
        d.Parse(json);

        auto tno = [&] (auto const &s) -> decltype(auto) {
#if 0
            return session->translator->ut(s);
#else
            return s;
#endif
        };

        for (int i = 0; i < d.Size(); i++) {
            Value const &di = d[i];
            std::string cmd = di[0].GetString();
            const char *maybeNodeName = di.Size() >= 1 && di[1].IsString() ? di[1].GetString() : "(not a node)";
            GraphException::translated([&] {
                if (0) {
                } else if (cmd == "addNode") {
                    addNode(tno(di[1].GetString()), di[2].GetString());
                } else if (cmd == "completeNode") {
                    completeNode(di[1].GetString());
                } else if (cmd == "setNodeInput") {
                    setNodeInput(di[1].GetString(), tno(di[2].GetString()), generic_get<zany>(di[3]));
                } else if (cmd == "setNodeParam") {
                    setNodeParam(di[1].GetString(), tno(di[2].GetString()), generic_get<std::variant<int, float, std::string>, false>(di[3]));
                /*} else if (cmd == "setNodeOption") {
                    setNodeOption(di[1].GetString(), di[2].GetString());*/
                } else if (cmd == "bindNodeInput") {
                    bindNodeInput(di[1].GetString(), tno(di[2].GetString()), di[3].GetString(), tno(di[4].GetString()));
                } else if (cmd == "setBeginFrameNumber") {
                    this->beginFrameNumber = di[1].GetInt();
                } else if (cmd == "setEndFrameNumber") {
                    this->endFrameNumber = di[1].GetInt();
                } else {
                    log_warn("got unexpected command: {}", cmd);
                }
            }, maybeNodeName);
        }
    }, *session->globalStatus);
}

}
