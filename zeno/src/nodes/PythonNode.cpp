#include <zeno/zeno.h>

namespace zeno {
    namespace {
        struct PythonNode : zeno::INode {
            virtual void apply() override {
            }
        };

        ZENDEFNODE(PythonNode, {
            {{"string", "script"},{"list", "args"}},
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
            {{"string", "nameList"},{"string", "keyWords"},{"readpath", "materialPath"},{"string", "matchInputs"}, {"string", "script"}},
            {},
            {},
            {"command"},
            });
    }
}