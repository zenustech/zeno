#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include "reflect/core.hpp"
#include "reflect/type"
#include "reflect/reflection_traits.hpp"


namespace zeno
{
    struct ZRECORD() TestReflectNode : public zeno::INode
    {
        TestReflectNode() = default;

        int apply(std::string wtf, char c) {
            param_b = wtf;
            param_c = c;
            return 233;
        }

        virtual zeno::reflect::TypeHandle gettype() {
            return zeno::reflect::get_type<TestReflectNode>();
        }

        int param_a = 3;
        std::string param_b = "default";
        char param_c = 'c';
    };
}



