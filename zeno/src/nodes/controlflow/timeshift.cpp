#include <zeno/zeno.h>
#include <zeno/core/reflectdef.h>
#include "zeno_types/reflect/reflection.generated.hpp"


namespace zeno {

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
            //Êä³ö
            _Group {
                {"", ParamObject("Output Object")},
            }
        };

        std::shared_ptr<IObject> apply(zany prim, int offset) {
            return prim;
        }
    };
}

