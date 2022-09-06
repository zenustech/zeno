#include <zenomodel/include/api.h>

int main()
{
    Zeno_NewFile();
    ZENO_HANDLE hGraph = Zeno_GetGraph("main");
    ZENO_HANDLE hCube = Zeno_AddNode(hGraph, "CreateCube");
    ZENO_HANDLE hTrans = Zeno_AddNode(hGraph, "TransformPrimitive");

    ZENO_ERROR err = Zeno_AddLink(hCube, "prim", hTrans, "prim");
    Zeno_SaveAs("C:/zeno-api/abc.zsg");

    return 0;
}