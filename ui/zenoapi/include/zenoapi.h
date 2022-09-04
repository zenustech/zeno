#ifndef __ZENO_UI_API_H__
#define __ZENO_UI_API_H__

#include "interface.h"

namespace zenoapi
{
    ZENO_ERROR  NewScene();
    ZENO_HANDLE CreateGraph(const std::string& name);
    ZENO_ERROR  DeleteGraph(ZENO_HANDLE hSubgraph);
    ZENO_HANDLE GetGraph(const std::string& name);
    ZENO_ERROR  RenameGraph(ZENO_HANDLE hSubgraph, const std::string& newName);
    int GetCount();
    ZENO_HANDLE GetItem(int idx);

    ZENO_ERROR  AddNode(ZENO_HANDLE hGraph, const std::string& nodeCls);
    ZENO_HANDLE GetNode(const std::string& ident);
    ZENO_ERROR  DeleteNode(ZENO_HANDLE hNode);

    //io
    ZENO_ERROR openFile(const std::string& fn);
    ZENO_ERROR saveFile(const std::string& fn);
}

#endif