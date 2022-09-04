#include "zenoapi.h"
#include "zapplication_impl.h"

namespace zenoapi
{
    ZENO_ERROR NewScene()
    {
        return 0;
    }

    ZENO_HANDLE CreateGraph(const std::string &name)
    {
        return 0;
    }

    ZENO_ERROR DeleteGraph(ZENO_HANDLE hSubgraph)
    {
        return 0;
    }

    ZENO_HANDLE GetGraph(const std::string &name)
    {
        return 0;
    }

    ZENO_ERROR RenameGraph(ZENO_HANDLE hSubgraph, const std::string &newName)
    {
        return 0;
    }

    int GetCount()
    {
        return 0;
    }

    ZENO_HANDLE GetItem(int idx)
    {
        return 0;
    }

    ZENO_ERROR AddNode(ZENO_HANDLE hGraph, const std::string &nodeCls)
    {
        return 0;
    }

    ZENO_HANDLE GetNode(const std::string &ident)
    {
        return 0;
    }

    ZENO_ERROR DeleteNode(ZENO_HANDLE hNode) {
        return 0;
    }

    //io
    ZENO_ERROR openFile(const std::string &fn) {
        return 0;
    }

    ZENO_ERROR saveFile(const std::string &fn) {
        return 0;
    }
}
