#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>


using namespace zeno;

namespace zeno{
/**
 * @brief 该节点是用来debug的。用于读取cvs格式的物理场
 * 
 */
struct readCsv: INode
{

    virtual void apply() override
    {
        auto prim = get_input<PrimitiveObject>("prim");
        
        
        set_output("neighborList", std::move(neighborList));
    }
};

ZENDEFNODE(readCsv,
    {
        {

        },
        {""},
        {},
        {"PBD"},
    }
);


}//zeno
