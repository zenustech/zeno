#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Graph.h>
#include <zeno/types/ListObject.h>
#include <zeno/core/reflectdef.h>
#include "zeno_types/reflect/reflection.generated.hpp"


namespace zeno
{
    struct ZDEFNODE() Switch : INode
    {
        ReflectCustomUI m_uilayout = {
            _ObjectGroup {{{"input_objects", "Input Objects", Socket_Clone},}},
            //以下填的是以参数形式返回的外部引用
            {},
            //返回值信息：
            _ObjectParam {"", "Output Object", Socket_Output},
            {}, {}
        };

        std::shared_ptr<IObject> apply(std::shared_ptr<ListObject> input_objects, int switch_num) {
            if (!input_objects) {
                throw makeError<UnimplError>("no input objects on Switch");
            }
            int n = input_objects->size();
            int clip_switch = std::min(n - 1, std::max(0, switch_num));
            zany obj = input_objects->get(clip_switch);
            return obj;
        }
    };

    //TODO:
    //struct ZDEFNODE() SwitchIf : INode
    //{
    //};
}
