#include <zeno/zeno.h>

namespace zeno {
    namespace {
        struct PythonNode : zeno::INode {
            virtual void apply() override {

            }
        };

        ZENDEFNODE(PythonNode, {
            {{"string", "script"}},
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
    }
}