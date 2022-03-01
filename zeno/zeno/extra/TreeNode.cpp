#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/types/NumericObject.h>
#include <cassert>

namespace zeno {

ZENO_API TreeNode::TreeNode() = default;
ZENO_API TreeNode::~TreeNode() = default;

ZENO_API void TreeNode::apply() {
    auto tree = std::make_shared<TreeObject>(this);
    set_output("out", std::move(tree));
}

ZENO_API std::string EmissionPass::finalizeCode() {
    auto defs = collectDefs();
    for (auto const &var: variables) {
        var.node->emitCode(this);
    }
    auto code = collectCode();
    code = defs + code;
    return {code};
}

ZENO_API int EmissionPass::determineType(IObject *object) {
    if (auto num = dynamic_cast<NumericObject *>(object)) {
        return std::visit([&] (auto const &value) -> int {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<float, T>) {
                return 1;
            } else if constexpr (std::is_same_v<vec2f, T>) {
                return 2;
            } else if constexpr (std::is_same_v<vec3f, T>) {
                return 3;
            } else if constexpr (std::is_same_v<vec4f, T>) {
                return 4;
            } else {
                throw zeno::Exception("bad numeric object type: " + (std::string)typeid(T).name());
            }
        }, num->value);

    } else if (auto tree = dynamic_cast<TreeObject *>(object)) {
        assert(tree->node);
        if (auto it = varmap.find(tree->node); it != varmap.end())
            return variables.at(it->second).type;
        int type = tree->node->determineType(this);
        varmap[tree->node] = variables.size();
        variables.push_back(VarInfo{type, tree->node});
        return type;

    } else {
        throw zeno::Exception("bad tree object type: " + (std::string)typeid(*object).name());
    }
}

ZENO_API std::string EmissionPass::typeNameOf(int type) {
    if (type == 1) return "float";
    else return "vec" + std::to_string(type);
}

ZENO_API std::string EmissionPass::collectDefs() const {
    std::string res;
    int cnt = 0;
    for (auto const &var: variables) {
        res += typeNameOf(var.type) + " tmp" + std::to_string(cnt) + ";\n";
        cnt++;
    }
    return res;
}

ZENO_API std::string EmissionPass::collectCode() const {
    std::string res;
    for (auto const &line: lines) {
        res += line + "\n";
    }
    return res;
}

ZENO_API std::string EmissionPass::determineExpr(IObject *object) const {
    if (auto num = dynamic_cast<NumericObject *>(object)) {
        return std::visit([&] (auto const &value) -> std::string {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<float, T>) {
                return "float(" + std::to_string(value) + ")";
            } else if constexpr (std::is_same_v<vec2f, T>) {
                return "vec2(" + std::to_string(value[0]) + ", " + std::to_string(value[1]) + ")";
            } else if constexpr (std::is_same_v<vec3f, T>) {
                return "vec3(" + std::to_string(value[0]) + ", " + std::to_string(value[1]) + ", "
                    + std::to_string(value[2]) + ")";
            } else if constexpr (std::is_same_v<vec4f, T>) {
                return "vec4(" + std::to_string(value[0]) + ", " + std::to_string(value[1]) + ", "
                    + std::to_string(value[2]) + ", " + std::to_string(value[3]) + ")";
            } else {
                throw zeno::Exception("bad numeric object type: " + (std::string)typeid(T).name());
            }
        }, num->value);

    } else if (auto tree = dynamic_cast<TreeObject *>(object)) {
        return "tmp" + std::to_string(varmap.at(tree->node));
    }
}

ZENO_API void EmissionPass::emitCode(std::string const &line) {
    int idx = lines.size();
    lines.push_back("tmp" + std::to_string(idx) + " = " + line + ";");
}

ZENO_API std::string EmissionPass::finalizeCode(std::vector<std::string> const &keys, std::vector<std::shared_ptr<IObject>> const &vals) {
    std::map<std::string, int> vartypes;
    for (int i = 0; i < keys.size(); i++) {
        vartypes.emplace(keys[i], determineType(vals[i].get()));
    }
    auto code = finalizeCode();
    for (int i = 0; i < keys.size(); i++) {
        auto type = vartypes.at(keys[i]);
        auto expr = determineExpr(vals[i].get());
        code += typeNameOf(type) + " " + keys[i] + " = " + expr + ";";
    }
    return code;
}

}
