#include <zeno/zeno.h>
#include <zeno/core/defNode.h>


namespace zeno {

    struct TimeShift2 : INode {

        void apply() override {
            auto prim = clone_input("prim");
            set_output("prim", std::move(prim));
        }
    };

    ZENDEFNODE(TimeShift2, {
        {/*inputs:*/
            ParamObject("prim", gParamType_Geometry),
            ParamPrimitive("offset", gParamType_Int),
            ParamPrimitive("clamp", gParamType_String, "None", zeno::Combobox, std::vector<std::string>{"None", "Clamp to First", "Clamp to Last", "Clamp to Both"}),
            ParamPrimitive("start frame", gParamType_Int, 0, Lineedit, Any(), "visible = parameter('clamp').value == 'Clamp to First' || parameter('clamp').value == 'Clamp to Both';"),
            ParamPrimitive("end frame", gParamType_Int, 0, Lineedit, Any(), "visible = parameter('clamp').value == 'Clamp to Last' || parameter('clamp').value == 'Clamp to Both';")
        },
        {/*outputs:*/
            {gParamType_Geometry, "prim"}
        },
        {},
        {"create"},
    });

#if 0
    struct ZDEFNODE() TimeShift : INode {

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "clamp", Control = zeno::Combobox, ComboBoxItems = ("None", "Clamp to First", "Clamp to Last", "Clamp to Both"))
            std::string m_clamp = "None";

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "start frame", Constrain = "visible = parameter('clamp').value == 'Clamp to First' || parameter('clamp').value == 'Clamp to Both';")
            int m_startFrame = 0;

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "end frame", Constrain = "visible = parameter('clamp').value == 'Clamp to Last' || parameter('clamp').value == 'Clamp to Both';")
            int m_endFrame = 0;

        ReflectCustomUI m_uilayout = {
            _Group {
                {"prim", ParamObject("prim", Socket_Clone)},
                {"offset", ParamPrimitive("offset")},
            },
            _Group {
                {"", ParamObject("Output Object")},
            }
        };

        std::shared_ptr<IObject> apply(zany prim, int offset) {
            return prim;
        }
    };
#endif
}

