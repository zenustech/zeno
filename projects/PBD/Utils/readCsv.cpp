#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "../Utils/readFile.h"

using namespace zeno;

namespace zeno{
/**
 * @brief 该节点是用来debug的。用于读取cvs格式的物理场到pos中。
 * 
 */
struct readCsv: INode
{

    virtual void apply() override
    {
        auto prim = get_input<PrimitiveObject>("prim");
        auto path = get_input<StringObject>("path")->get();

        auto &pos = prim->verts;
        pos.clear();
        readVectorField(path,pos);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(readCsv,
    {
        {"path"},
        {gParamType_Primitive, "prim"},
        {},
        {"PBD"},
    }
);


}//zeno
