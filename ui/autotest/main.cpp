#include <zenoapi/include/zenoapi.h>

int main()
{
    Zeno_NewFile();
    ZENO_HANDLE hGraph = Zeno_GetGraph("main");
    ZENO_HANDLE hCube = Zeno_AddNode(hGraph, "CreateCube");
    ZENO_HANDLE hTrans = Zeno_AddNode(hGraph, "TransformPrimitive");

    ZENO_ERROR err = Zeno_AddLink(hCube, "prim", hTrans, "prim");
    //err = Zeno_RemoveLink(hCube, "prim", hTrans, "prim");

    Zeno_SaveFile("C:/zeno-api/abc.zsg");

    //std::string ident;
    //Zeno_GetIdent(hNode, ident);
    //ZENO_HANDLE hNode2 = Zeno_GetNode(ident);
    //ZENO_HANDLE hNode3 = Zeno_GetNode(ident);

    return 0;
}