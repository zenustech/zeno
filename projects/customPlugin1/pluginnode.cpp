#include <zeno/zeno.h>
#include <zeno/core/reflectdef.h>
#include "plugindef.h"
#include "custom_plugin1/reflect/custom_plugin1/plugindef.h.generated.hpp"


namespace zeno
{
    /*
    struct CustomPlugin1Node : zeno::INode {
        virtual void apply() override {
            auto path = get_input<StringObject>("path")->get(); // std::string
            std::shared_ptr<PrimitiveObject> spObj = std::make_shared<PrimitiveObject>();
            //auto result = zeno::readWav(path);
            set_output("prim", std::move(spObj)); 
        }
    };

    ZENDEFNODE(CustomPlugin1Node, {
        {
            {zeno::types::gParamType_String, "path", "", zeno::Socket_Primitve, zeno::ReadPathEdit},
        },
        {
            {gParamType_Primitive, "prim"},
        },
        {},
        {
            "customplugin1"
        },
    });
    */

    struct ZDEFNODE(DisplayName = "IsImplNode2") PluginRefNode : zeno::INode
    {
        /*
        ReflectCustomUI m_uilayout = {
                _ObjectGroup {
                    {
                        _ObjectParam {"input_obj", "Input Object", Socket_Owning},
                    }
                },
                _ObjectGroup {
                    {
                        //空字符串默认mapping到 apply的输出值
                        _ObjectParam {"output_obj", "Output Object", Socket_Output},
                    }
               },
                _ParamTab {
                    "Tab1",
                    {
                        _ParamGroup {
                            "Group1",
                            {
                                _Param { "wtf", "ABC", std::string("a1") },
                                _Param { "c", "input vector", zeno::vec3f({ 0,1,0 }) },
                            }
                        },
                    }
                },
                _ParamGroup {

                }
        };
        */

        std::shared_ptr<zeno::PrimitiveObject> apply(std::shared_ptr<zeno::PrimitiveObject> input_obj, std::string wtf = "abc", zeno::vec3f c = zeno::vec3f({ 0,1,0 })/*, float& ret1, std::shared_ptr<zeno::IObject>&output_obj*/) {
            //ret1 = 8;
            zeno::reflect::Any vec = zeno::reflect::make_any<zeno::vec3f>(zeno::vec3f({ 0.,1.0,2. }));
            zeno::reflect::any_cast<zeno::vec3f>(vec);
            vec.type();

            std::shared_ptr<zeno::PrimitiveObject> resObj = std::make_shared<zeno::PrimitiveObject>();
            std::shared_ptr<zeno::IObject> baseObj = resObj;

            bool bConverted = std::is_convertible<int, float>::value;
            bConverted = std::is_convertible<std::shared_ptr<PrimitiveObject>, std::shared_ptr<LightObject>>::value;

            return resObj;
            //return input_obj;     //TODO: 直接返回这个，外部的res会空（INode.cpp: iter->second.spObject = zeno::reflect::any_cast<std::shared_ptr<zeno::IObject>>(res);)
        }
    };

    struct ZDEFNODE() MakeCustomPluginData : zeno::INode
    {
        CustomDefPrimitive1 apply(int c) {
            CustomDefPrimitive1 dat;
            dat.m1 = { 2,3,1 };
            dat.m3 = 235;
            return dat;
        }
    };

    struct ZDEFNODE() PluginCustomDataNode : zeno::INode
    {
        CustomDefPrimitive1 apply(CustomDefPrimitive1 data1, zeno::vec3f cc = zeno::vec3f({ 1,2,3 })) {
            return data1;
        }
    };
}
