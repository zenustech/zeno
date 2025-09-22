﻿#include <zeno/core/Graph.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <zeno/funcs/LiterialConverter.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/extra/GraphException.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/core/Session.h>
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
        //log_warn("unknown type encountered in generic_get");
        return cast(0);
    }
}

ZENO_API void Graph::loadGraph(const char *json) {
    Document d;
    d.Parse(json);

    if (!d.IsArray()) {
        throw GraphException { "None", nullptr };
    }

    Graph *g = this;
    std::stack<Graph *> gStack;

    for (int i = 0; i < d.Size(); i++) {
        Value const &di = d[i];
        std::string cmd = di[0].GetString();
        const char *maybeNodeName = cmd == "addNode" ? di[2].GetString() : (
            di.Size() >= 1 && di[1].IsString() ? di[1].GetString() : "(not a node)");
        //ZENO_P(cmd);
        //ZENO_P(maybeNodeName);
        GraphException::translated([&] {
            if (0) {
            } else if (cmd == "addNode") {
                g->addNode(di[1].GetString(), di[2].GetString());
            } else if (cmd == "setNodeInput") {
                g->setNodeInput(di[1].GetString(), di[2].GetString(), generic_get<zany>(di[3]));
            } else if (cmd == "setKeyFrame") {
                g->setKeyFrame(di[1].GetString(), di[2].GetString(), generic_get<zany>(di[3]));
            } else if (cmd == "setFormula") {
                g->setFormula(di[1].GetString(), di[2].GetString(), generic_get<zany>(di[3]));
            } else if (cmd == "setNodeParam") {
                g->setNodeParam(di[1].GetString(), di[2].GetString(), generic_get<std::variant<int, float, std::string, zany>, false>(di[3]));
            } else if (cmd == "bindNodeInput") {
                g->bindNodeInput(di[1].GetString(), di[2].GetString(), di[3].GetString(), di[4].GetString());
            } else if (cmd == "completeNode") {
                g->completeNode(di[1].GetString());
            } else if (cmd == "addSubnetNode") {
                auto newG = g->addSubnetNode(/*di[1].GetString(), */di[2].GetString());
            } else if (cmd == "addNodeOutput") {
                g->addNodeOutput(di[1].GetString(), di[2].GetString());
            } else if (cmd == "pushSubnetScope") {
                gStack.push(g);
                g = g->getSubnetGraph(di[1].GetString());
            } else if (cmd == "popSubnetScope") {
                g = gStack.top();
                gStack.pop();
            } else if (cmd == "setBeginFrameNumber") {
                this->beginFrameNumber = di[1].GetInt();
            } else if (cmd == "setEndFrameNumber") {
                this->endFrameNumber = di[1].GetInt();
            } else if (cmd == "setNodeOption") {
                // skip this for compatibility
            } else if (cmd == "markNodeChanged") {
                auto ident = di[1].GetString();
                auto &dc = g->getDirtyChecker();
                dc.taintThisNode(ident);
                //todo: mark node data change.
            } else if (cmd == "cacheToDisk") {
                g->setTempCache(di[1].GetString());
            } else if (cmd == "objRunType") {
                g->setObjRunType(di[1].GetString(), di[2].GetString());
            } else if (cmd == "viewId") {
                auto ident = std::string(di[1].GetString());
                zeno::getSession().globalComm->allViewNodes += ident + ' ';
            } else if (cmd == "enableTimer") {
                g->setEnableTimer(di[1].GetString());
            } else {
                log_warn("got unexpected command: {}", cmd);
            }
        }, maybeNodeName);
    }
}

}
