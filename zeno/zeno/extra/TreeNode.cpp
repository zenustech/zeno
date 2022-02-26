#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/types/NumericObject.h>
#include <cassert>

namespace zeno {

ZENO_API TreeNode::TreeNode() = default;
ZENO_API TreeNode::~TreeNode() = default;

ZENO_API void TreeNode::apply() {
    auto tree = std::make_shared<TreeObject>();
    tree->node = shared_from_this();
    set_output(output, std::move(tree));
}

ZENO_API int TreeNode::determineTypeOf(IObject *object) {
    if (auto num = dynamic_cast<NumericObject *>(object)) {
        return std::visit([&] (auto const &value) -> int {
            using T = std::decay_t<decltype(value)>;
            constexpr int N = zeno::is_vec_n<T>;
            return N;
        }, num->value);
    } else if (auto tree = dynamic_cast<TreeObject *>(object)) {
        assert(tree->node);
        return tree->node->determineType();
    } else {
        throw zeno::Exception("bad tree object type: " + typeid(*object).name());
    }
}

}
