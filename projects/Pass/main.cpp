#include <zeno/zeno.h>

namespace zeno {
namespace {

struct ForwardPass : INode {
    virtual void apply() override {
    }
};

ZENO_DEFNODE(ForwardPass)({
});

}
}
