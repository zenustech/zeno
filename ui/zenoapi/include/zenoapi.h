#ifndef __ZENO_UI_API_H__
#define __ZENO_UI_API_H__

#include "interface.h"

//io
ZENO_ERROR Zeno_NewFile();
ZENO_ERROR Zeno_OpenFile(const std::string &fn);
ZENO_ERROR Zeno_saveFile(const std::string &fn);

ZENO_HANDLE Zeno_CreateGraph(const std::string& name);
ZENO_ERROR  Zeno_DeleteGraph(ZENO_HANDLE hSubgraph);
ZENO_HANDLE Zeno_GetGraph(const std::string& name);
ZENO_ERROR  Zeno_RenameGraph(ZENO_HANDLE hSubgraph, const std::string& newName);
int Zeno_GetCount();
ZENO_HANDLE Zeno_GetItem(int idx);

ZENO_HANDLE Zeno_AddNode(ZENO_HANDLE hGraph, const std::string &nodeCls);
ZENO_HANDLE Zeno_GetNode(const std::string& ident);
ZENO_ERROR  Zeno_DeleteNode(ZENO_HANDLE hNode);
ZENO_ERROR  Zeno_GetName(ZENO_HANDLE hNode, std::string& name);
ZENO_ERROR  Zeno_GetIdent(ZENO_HANDLE hNode, std::string& ident);

#endif