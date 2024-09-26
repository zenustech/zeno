#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Graph.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/core/reflectdef.h>
#include "zeno_types/reflect/reflection.generated.hpp"


namespace zeno
{
    struct ZDEFNODE() Duplicate : INode
    {
        ReflectCustomUI m_uilayout = {
            // ‰»Î£∫
            _Group {
                {"input_object", ParamObject("Original Object", Socket_Clone)},
                {"keepModifyOriginal", ParamPrimitive("Keep Modify Original")},
            },
            // ‰≥ˆ£∫
            _Group {
                {"", ParamObject("Duplicated Object", Socket_Output)},
                {"", ParamObject("Original Object", Socket_Output, "", "visible = param('Keep Modify Original').value == 1;")},
            }
        };

        std::pair<zany, zany> apply(zany input_object, bool keepModifyOriginal)
        {
            auto res = std::make_pair<zany, zany>(nullptr, nullptr);
            res.first = input_object->clone();
            if (keepModifyOriginal) {
                res.second = input_object;
            }
            return res;
        }
    };

    struct ZDEFNODE() Duplicate3 : INode
    {
        std::tuple<zany, zany, int> apply(zany input_object, bool keepModifyOriginal)
        {
            return std::make_tuple<zany, zany, int>(nullptr, nullptr, 0);
        }
    };
}