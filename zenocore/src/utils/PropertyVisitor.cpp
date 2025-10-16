
#include "zeno/utils/PropertyVisitor.h"

#if 0

zeno::reflect::NodeParameterBase::NodeParameterBase(zeno::NodeImpl *Node) : Target(Node) {}

zeno::reflect::NodeParameterBase::~NodeParameterBase() = default;

zeno::reflect::NodeParameterBase::NodeParameterBase(zeno::reflect::NodeParameterBase &&RhsToMove) noexcept {
    Target = RhsToMove.Target;
    HookList = std::move(RhsToMove.HookList);

    RhsToMove.Target = nullptr;
}

void zeno::reflect::NodeParameterBase::RunInputHooks() const {
    if (nullptr != Target) {
        for (const auto& Hook : HookList.InputHook) {
            Hook(Target);
        }
    }
}

void zeno::reflect::NodeParameterBase::RunOutputHooks() const {
    if (nullptr != Target) {
        for (const auto& Hook : HookList.OutputHook) {
            Hook(Target);
        }
    }
}

void zeno::reflect::NodeParameterBase::RunBindingHooks() const {
    for (const auto& Hook : HookList.BindingHook) {
        Hook();
    }
}

#endif