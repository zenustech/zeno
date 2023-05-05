#include <zenomodel/include/api.h>

int main()
{
    Zeno_NewFile();
    ZENO_HANDLE hGraph = Zeno_GetGraph("main");
    ZENO_HANDLE hCube = Zeno_AddNode(hGraph, "CreateCube");
    ZENO_HANDLE hTrans = Zeno_AddNode(hGraph, "TransformPrimitive");

    Zeno_SetInputDefl(hGraph, hCube, "position", zeno::vec3f(2, 0, 0));
    ZENO_ERROR err = Zeno_AddLink(hGraph, hCube, "prim", hTrans, "prim");

    return 0;
}