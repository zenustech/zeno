#include <zenoapi/include/zenoapi.h>

int main()
{
    Zeno_NewFile();
    ZENO_HANDLE hGraph = Zeno_GetGraph("main");
    ZENO_HANDLE hNode = Zeno_AddNode(hGraph, "CreateCube");

    std::string ident;
    Zeno_GetIdent(hNode, ident);

    ZENO_HANDLE hNode2 = Zeno_GetNode(ident);
    ZENO_HANDLE hNode3 = Zeno_GetNode(ident);

    return 0;
}