#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/core/reflectdef.h>
#include <zeno/formula/zfxexecute.h>
#include <zeno/core/FunctionManager.h>
#include <zeno/types/GeometryObject.h>
#include "zeno_types/reflect/reflection.generated.hpp"


namespace zeno
{
    struct ZDEFNODE() AttributeWrangle : zeno::INode
    {
        ReflectCustomUI m_uilayout = {
            //输入：
            _Group {
                {"prim", ParamObject("Input Object", Socket_Clone)},
                {"zfxCode", ParamPrimitive("Zfx Code", "", CodeEditor)},
            },
            //输出：
            _Group {
                {"", ParamObject("Output Object")},
            }
        };

        std::shared_ptr<PrimitiveObject> apply(std::shared_ptr<PrimitiveObject> prim, std::string zfxCode) {
            std::shared_ptr<GeometryObject> out_prim = std::make_shared<GeometryObject>(prim.get());

            ZfxContext ctx;
            ctx.spNode = shared_from_this();
            ctx.spObject = out_prim;
            ctx.code = zfxCode;
            ZfxExecute zfx(zfxCode, &ctx);
            zfx.execute();

            if (auto spGeo = std::dynamic_pointer_cast<GeometryObject>(ctx.spObject)) {
                return spGeo->toPrimitive();
            }
            else {
                return prim;
            }
        }
    };
}