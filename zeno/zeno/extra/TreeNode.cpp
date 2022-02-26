#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/types/NumericObject.h>

namespace zeno {

ZENO_API void TreeNode::settle(std::string output,
                               std::vector<std::string> const &inputs,
                               std::vector<std::string> const &params) {
    auto tree = std::make_shared<TreeObject>();
    for (auto const &in: inputs) {
        tree->inputs.push_back(get_input(in));
    }
    for (auto const &key: params) {
        tree->params[key] = get_input(key);
    }
    set_output(output, std::move(tree));
}

}
