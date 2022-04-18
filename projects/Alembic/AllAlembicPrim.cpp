/* auto generated from: D:/AllAlembicPrim.zsg */
#include <zeno/zeno.h>
#include <zeno/extra/ISubgraphNode.h>
namespace {
struct AllAlembicPrim : zeno::ISerialSubgraphNode {
    virtual const char *get_subgraph_json() override {
        return R"ZSL(
[["addNode", "PrimMerge", "03e97822-PrimMerge"], ["bindNodeInput", "03e97822-PrimMerge", "listPrim", "e353bfe3-EndForEach", "list"], ["setNodeInput", "03e97822-PrimMerge", "tagAttr", ""], ["setNodeOption", "03e97822-PrimMerge", "VIEW"], ["completeNode", "03e97822-PrimMerge"], ["addNode", "GetAlembicPrim", "6f84912c-GetAlembicPrim"], ["bindNodeInput", "6f84912c-GetAlembicPrim", "abctree", "c3ac385d-SubInput", "port"], ["bindNodeInput", "6f84912c-GetAlembicPrim", "index", "121ac882-BeginFor", "index"], ["bindNodeInput", "6f84912c-GetAlembicPrim", "use_xform", "3cf59fc9-SubInput", "port"], ["completeNode", "6f84912c-GetAlembicPrim"], ["addNode", "SubInput", "c3ac385d-SubInput"], ["setNodeParam", "c3ac385d-SubInput", "name", "abctree"], ["setNodeParam", "c3ac385d-SubInput", "type", ""], ["setNodeParam", "c3ac385d-SubInput", "defl", ""], ["completeNode", "c3ac385d-SubInput"], ["addNode", "SubCategory", "3d0a8734-SubCategory"], ["setNodeParam", "3d0a8734-SubCategory", "name", "alembic"], ["completeNode", "3d0a8734-SubCategory"], ["addNode", "EndForEach", "e353bfe3-EndForEach"], ["bindNodeInput", "e353bfe3-EndForEach", "object", "6f84912c-GetAlembicPrim", "prim"], ["setNodeInput", "e353bfe3-EndForEach", "accept", true], ["bindNodeInput", "e353bfe3-EndForEach", "FOR", "121ac882-BeginFor", "FOR"], ["setNodeParam", "e353bfe3-EndForEach", "doConcat", false], ["completeNode", "e353bfe3-EndForEach"], ["addNode", "SubOutput", "3b640f2e-SubOutput"], ["bindNodeInput", "3b640f2e-SubOutput", "port", "03e97822-PrimMerge", "prim"], ["setNodeParam", "3b640f2e-SubOutput", "name", "prim"], ["setNodeParam", "3b640f2e-SubOutput", "type", ""], ["setNodeParam", "3b640f2e-SubOutput", "defl", ""], ["completeNode", "3b640f2e-SubOutput"], ["addNode", "SubInput", "3cf59fc9-SubInput"], ["setNodeParam", "3cf59fc9-SubInput", "name", "use_xform"], ["setNodeParam", "3cf59fc9-SubInput", "type", "int"], ["setNodeParam", "3cf59fc9-SubInput", "defl", "0"], ["completeNode", "3cf59fc9-SubInput"], ["addNode", "BeginFor", "121ac882-BeginFor"], ["bindNodeInput", "121ac882-BeginFor", "count", "cc278d89-CountAlembicPrims", "count"], ["completeNode", "121ac882-BeginFor"], ["addNode", "CountAlembicPrims", "cc278d89-CountAlembicPrims"], ["bindNodeInput", "cc278d89-CountAlembicPrims", "abctree", "c3ac385d-SubInput", "port"], ["completeNode", "cc278d89-CountAlembicPrims"]]
)ZSL";
    }
};
ZENDEFNODE(AllAlembicPrim, {
    {{"", "abctree", ""}, {"int", "use_xform", "0"}},
    {{"", "prim", ""}},
    {},
    {"alembic"},
});
}
