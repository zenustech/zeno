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
        int apply(std::string wtf, zeno::vec3f c) {
            param_b = wtf;
            return 233;
        }

        virtual std::shared_ptr<zeno::reflect::TypeHandle> getReflectType() override {
            std::shared_ptr<zeno::reflect::TypeHandle> m_handle = std::make_shared<zeno::reflect::TypeHandle>(zeno::reflect::get_type<TestReflectNode>());
            return m_handle;
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

    struct ZRECORD() SimpleReflect : public zeno::INode
    {
        SimpleReflect() = default;

        std::string apply(std::shared_ptr<zeno::IObject> input_obj, std::string wtf, zeno::vec3f c, float& ret1, std::shared_ptr<zeno::IObject>& output_obj) {
            ret1 = 8;
            return "";
        }

        virtual std::shared_ptr<zeno::reflect::TypeHandle> getReflectType() override {
            std::shared_ptr<zeno::reflect::TypeHandle> m_handle = std::make_shared<zeno::reflect::TypeHandle>(zeno::reflect::get_type<SimpleReflect>());
            return m_handle;
        };
    };

    struct ZRECORD() ReadOnlyNode : public zeno::INode
    {
        virtual std::shared_ptr<zeno::reflect::TypeHandle> getReflectType() override {
            std::shared_ptr<zeno::reflect::TypeHandle> m_handle = std::make_shared<zeno::reflect::TypeHandle>(zeno::reflect::get_type<ReadOnlyNode>());
            return m_handle;
        };

        std::shared_ptr<const zeno::IObject> apply(std::shared_ptr<const zeno::IObject> input_obj, const std::string& name1, const std::string& name2, std::string & ret)
        {
            ret = input_obj->nodeId;
            return input_obj;
        }
    };
}