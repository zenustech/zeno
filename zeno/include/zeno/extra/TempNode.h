#pragma once

#include <string>
#include <memory>
#include <zeno/core/Graph.h>
#include <zeno/core/IObject.h>
#include <zeno/core/INode.h>
#include <zeno/funcs/LiterialConverter.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/utils/safe_at.h>

namespace zeno {

struct TempNodeCaller {
    Graph *graph;
    std::string nodety;
    bool called;
    std::map<std::string, std::shared_ptr<IObject>> params;

    explicit TempNodeCaller(Graph *graph, std::string const &nodety)
        : graph(graph), nodety(nodety), called(false)
    {}

    TempNodeCaller &set(std::string const &id, std::shared_ptr<IObject> obj) {
        params[id] = std::move(obj);
        return *this;
    }

    template <class T>
    TempNodeCaller &set2(std::string const &id, T const &val) {
        params[id] = objectFromLiterial(val);
        return *this;
    }

    std::shared_ptr<IObject> get(std::string const &sockid) {
        if (!called) {
            params = graph->callTempNode(nodety, params);
            called = true;
        }
        return safe_at(params, sockid, "output socket of temp node `" + nodety + "`");
    }

    template <class T>
    std::shared_ptr<T> get(std::string const &sockid) {
        return safe_dynamic_cast<T>(get(sockid), "output socket `" + sockid + "` of temp node `" + nodety + "`");
    }

    template <class T>
    T get2(std::string const &sockid) {
        return objectToLiterial<T>(get(sockid), "output socket `" + sockid + "` of temp node `" + nodety + "`");
    }
};

}
