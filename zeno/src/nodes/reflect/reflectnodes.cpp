#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <memory>
#include "reflect/core.hpp"
#include "reflect/type"
#include "reflect/reflection_traits.hpp"
#include "reflect/reflection.generated.hpp"


#define ZENO_REFLECT_TYPE(T) \
virtual std::shared_ptr<zeno::reflect::TypeHandle> getReflectType() override {\
    return std::make_shared<zeno::reflect::TypeHandle>(zeno::reflect::get_type<T>());\
}

namespace zeno
{
    struct ZRECORD() TestReflectNode : zeno::INode
    {
        TestReflectNode() = default;

        ZENO_REFLECT_TYPE(TestReflectNode)

        ZMETHOD(Name = "×öÐ©ÊÂ")
        int apply(std::string wtf, zeno::vec3f c) {
            param_b = wtf;
            return 233;
        }

        ZPROPERTY(Role = zeno::Role_InputObject, DisplayName = "Input Object", Socket = zeno::Socket_Owning)
        std::shared_ptr<zeno::IObject> m_prim;

        ZPROPERTY(Role = zeno::Role_OutputObject, DisplayName = "Output Object")
        std::shared_ptr<zeno::IObject> m_result;

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "Param A")
        int param_a = 3;

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName="Param B", Control = zeno::Multiline)
        std::string param_b = "default";

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "Param Options", Control = zeno::Combobox, ComboBoxItems = ("option A", "option B"))
        std::string param_options = "Option A";

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "Dt", Control = zeno::Slider, range = (1, 100, 1))
        int dt = 0;

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "Position")
        zeno::vec3f pos;

        ZPROPERTY(Role = zeno::Role_OutputPrimitive)
        zeno::vec3f outvec;
    };

    struct ZRECORD() SimpleReflect : zeno::INode
    {
        SimpleReflect() = default;

        ZENO_REFLECT_TYPE(SimpleReflect)

        std::string apply(std::shared_ptr<zeno::PrimitiveObject> input_obj, std::string wtf, zeno::vec3f c, float& ret1, std::shared_ptr<zeno::IObject>& output_obj) {
            ret1 = 8;
            return "";
        }
    };

    struct ZRECORD() ReadOnlyNode : zeno::INode
    {
        ZENO_REFLECT_TYPE(TestReflectNode)

        std::shared_ptr<const zeno::IObject> apply(std::shared_ptr<const zeno::IObject> input_obj, const std::string& name1, const std::string& name2, std::string & ret)
        {
            ret = input_obj->nodeId;
            return input_obj;
        }
    };
}