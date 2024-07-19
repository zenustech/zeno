#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include "reflect/core.hpp"
#include "reflect/type"
#include "reflect/reflection_traits.hpp"
#include "reflect/reflection.generated.hpp"


namespace zeno
{
    struct ZRECORD() TestReflectNode : public zeno::INode
    {
        TestReflectNode() = default;

        ZMETHOD(Name = "×öÐ©ÊÂ")
        int apply(std::string wtf, char c) {
            param_b = wtf;
            param_c = c;
            return 233;
        }

        virtual zeno::reflect::TypeHandle gettype() {
            return zeno::reflect::get_type<TestReflectNode>();
        }

        ZPROPERTY(Role = "input", DisplayName = "Input Object")
        std::shared_ptr<IObject> m_prim;

        ZPROPERTY(Role = "input", DisplayName = "Param A")
        int param_a = 3;

        ZPROPERTY(Role = "input", DisplayName="Param B", Control = "Multiline")
        std::string param_b = "default";

        ZPROPERTY(Role = "input", DisplayName = "Param Options", Control = "ComboBox"/*, ComboBoxItems = ("option A", "option B")*/)
        std::string param_options = "option A";

        ZPROPERTY(Role = "input", DisplayName = "Position")
        zeno::vec3f pos;

        char param_c = 'c';
    };
}