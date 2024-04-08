#include <zeno/zeno.h>

namespace zeno {
    namespace {
        struct PythonNode : zeno::INode {
            virtual void apply() override {
            }
        };

        ZENDEFNODE(PythonNode, {
            {{"string", "script", "", ParamSocket, Multiline},{"list", "args"}},
            {},
            {},
            {"command"},
        });

        struct GenerateCommands : zeno::INode {
            virtual void apply() override {

            }
        };

        ZENDEFNODE(GenerateCommands, {
            {
                {"string", "source"},
                {"string", "commands"},
            },
            {},
            {},
            {"command"},
        });

        struct PythonMaterialNode : zeno::INode {
            virtual void apply() override {

            }
        };

        ZENDEFNODE(PythonMaterialNode, {
            {
                {"string", "nameList"},
                {"string", "keyWords"},
                {"string", "materialPath", "", zeno::ParamSocket, zeno::ReadPathEdit},
                {"string", "matchInputs"}, 
                {"string", "script", "", ParamSocket, Multiline}
            },
            {},
            {},
            {"command"},
            });
    }
}