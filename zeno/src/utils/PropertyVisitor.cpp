
#include "zeno/utils/PropertyVisitor.h"

zeno::reflect::NodeParameterBase::NodeParameterBase(zeno::INode *Node) : Target(Node) {}

zeno::reflect::NodeParameterBase::~NodeParameterBase() {
    for (const auto& Hook : HookList.OutputHook) {
        Hook(Target);
    }
}

zeno::reflect::NodeParameterBase::NodeParameterBase(zeno::reflect::NodeParameterBase &&RhsToMove) noexcept {
    Target = RhsToMove.Target;
    HookList = std::move(RhsToMove.HookList);

    RhsToMove.Target = nullptr;
}

void zeno::reflect::NodeParameterBase::RunInputHooks() const {
    for (const auto& Hook : HookList.InputHook) {
        Hook(Target);
    }
}
