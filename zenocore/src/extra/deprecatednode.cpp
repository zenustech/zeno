#include <zeno/zeno.h>


namespace zeno {
    namespace {

        struct DeprecatedNode : zeno::INode {
            virtual void apply() override {

            }
        };

        ZENDEFNODE(DeprecatedNode, {
            {},
            {},
            {},
            {"subgraph"}
        });

    }
}