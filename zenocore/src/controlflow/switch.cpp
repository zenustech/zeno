#include <zeno/zeno.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/utils/interfaceutil.h>


namespace zeno
{
    struct Switch : INode
    {
        void apply() override {
            auto input_objects = get_input_ListObject("Input Objects");
            if (!input_objects) {
                throw makeError<UnimplError>("no input objects on Switch");
            }
            int switch_num = get_input2_int("Switch Number");
            int n = input_objects->m_impl->size();
            int clip_switch = std::min(n - 1, std::max(0, switch_num));
            auto obj = input_objects->m_impl->get(clip_switch);
            set_output("Output Object", obj->clone());
        }
    };

    ZENDEFNODE(Switch,
    {
        {
            {gParamType_List, "Input Objects"},
            {gParamType_Int, "Switch Number"}
        },
        {
            {gParamType_Geometry, "Output Object"}
        },
        {},
        {"reflect"}
    });

    struct SwitchIf : INode
    {
        CustomUI export_customui() const override {
            CustomUI ui = INode::export_customui();
            ui.uistyle.iconResPath = ":/icons/node/switchif.svg";
            ui.uistyle.background = "#CEFFB3";
            return ui;
        }

        void apply() override {
            int cond = get_input2_int("Condition");
            if (cond != 0) {
                set_output("Output", clone_input("If True"));
            }
            else {
                set_output("Output", clone_input("If False"));
            }
        }
    };

    ZENDEFNODE(SwitchIf,
    {
        {
            {gParamType_IObject, "If False"},
            {gParamType_IObject, "If True"},
            {gParamType_Int, "Condition"}
        },
        {
            {gParamType_IObject, "Output"}
        },
        {},
        {"flow"}
    });

    struct SwitchBetween : INode
    {
        CustomUI export_customui() const override {
            CustomUI ui = INode::export_customui();
            ui.uistyle.iconResPath = ":/icons/node/switch-between.svg";
            ui.uistyle.background = "#CEFFB3";
            return ui;
        }

        void apply() override {
            int cond = get_input2_int("cond1");
            if (cond != 0) {
                set_output("Output", clone_input("b1"));
                return;
            }

            cond = get_input2_int("cond2");
            if (cond != 0) {
                set_output("Output", clone_input("b2"));
                return;
            }

            cond = get_input2_int("cond3");
            if (cond != 0) {
                set_output("Output", clone_input("b3"));
                return;
            }

            cond = get_input2_int("cond4");
            if (cond != 0) {
                set_output("Output", clone_input("b4"));
                return;
            }

            cond = get_input2_int("cond5");
            if (cond != 0) {
                set_output("Output", clone_input("b5"));
                return;
            }

            cond = get_input2_int("cond6");
            if (cond != 0) {
                set_output("Output", clone_input("b6"));
                return;
            }

            cond = get_input2_int("cond7");
            if (cond != 0) {
                set_output("Output", clone_input("b7"));
                return;
            }

            cond = get_input2_int("cond8");
            if (cond != 0) {
                set_output("Output", clone_input("b8"));
                return;
            }
        }
    };

    ZENDEFNODE(SwitchBetween,
    {
        {
            {gParamType_IObject, "b1"},
            {gParamType_IObject, "b2"},
            {gParamType_IObject, "b3"},
            {gParamType_IObject, "b4"},
            {gParamType_IObject, "b5"},
            {gParamType_IObject, "b6"},
            {gParamType_IObject, "b7"},
            {gParamType_IObject, "b8"},
            {gParamType_Int, "cond1", "0"},
            {gParamType_Int, "cond2", "0"},
            {gParamType_Int, "cond3", "0"},
            {gParamType_Int, "cond4", "0"},
            {gParamType_Int, "cond5", "0"},
            {gParamType_Int, "cond6", "0"},
            {gParamType_Int, "cond7", "0"},
            {gParamType_Int, "cond8", "0"},
        },
        {
            {gParamType_IObject, "Output"}
        },
        {},
        {"flow"}
    });
}
