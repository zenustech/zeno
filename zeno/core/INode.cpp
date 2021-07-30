#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/Session.h>
#include <zeno/types/ConditionObject.h>
#ifdef ZENO_VISUALIZATION
#include <zeno/extra/Visualization.h>
#endif
#ifdef ZENO_GLOBALSTATE
#include <zeno/extra/GlobalState.h>
#endif
#include <zeno/utils/safe_at.h>

namespace zeno {

ZENO_API INode::INode() = default;
ZENO_API INode::~INode() = default;

ZENO_API void INode::doComplete() {
    set_output("DST", std::make_shared<ConditionObject>());
    complete();
}

ZENO_API void INode::complete() {}

ZENO_API bool INode::checkApplyCondition() {
    /*if (has_input("COND")) {  // deprecated
        auto cond = get_input<zeno::ConditionObject>("COND");
        if (!cond->get())
            return false;
    }*/

#ifdef ZENO_GLOBALSTATE
    if (has_option("ONCE")) {  // TODO: frame control should be editor work
        if (!zeno::state.isFirstSubstep())
            return false;
    }

    if (has_option("PREP")) {
        if (!zeno::state.isOneSubstep())
            return false;
    }
#endif

    if (has_option("MUTE")) {
        auto desc = nodeClass->desc.get();
        if (desc->inputs[0].name != "SRC") {
            // TODO: MUTE should be an editor work
            muted_output = get_input(desc->inputs[0].name);
        } else {
            for (auto const &[ds, bound]: inputBounds) {
                muted_output = get_input(ds);
                break;
            }
        }
        return false;
    }

    return true;
}

ZENO_API void INode::doApply() {
    for (auto const &[ds, bound]: inputBounds) {
        requireInput(ds);
    }

    coreApply();
}

ZENO_API void INode::requireInput(std::string const &ds) {
    auto [sn, ss] = inputBounds.at(ds);
    graph->applyNode(sn);
    auto ref = graph->getNodeOutput(sn, ss);
    inputs[ds] = ref;
}

ZENO_API void INode::coreApply() {
    if (checkApplyCondition()) {
        apply();
    }

#ifdef ZENO_VISUALIZATION
    if (has_option("VIEW")) {
        graph->hasAnyView = true;
        if (!state.isOneSubstep())  // no duplicate view when multi-substep used
            return;
        if (!graph->isViewed)  // VIEW subnodes only if subgraph is VIEW'ed
            return;
        auto desc = nodeClass->desc.get();
        auto obj = muted_output ? muted_output
            : safe_at(outputs, desc->outputs[0].name, "output");
        auto path = Visualization::exportPath();
        obj->dumpfile(path);
    }
#endif
}

ZENO_API bool INode::has_option(std::string const &id) const {
    return options.find(id) != options.end();
}

ZENO_API bool INode::has_input(std::string const &id) const {
    return inputBounds.find(id) != inputBounds.end();
}

ZENO_API std::shared_ptr<IObject> INode::get_input(std::string const &id) const {
    return safe_at(inputs, id, "input", myname);
}

ZENO_API IValue INode::get_param(std::string const &id) const {
    return safe_at(params, id, "param", myname);
}

ZENO_API void INode::set_output(std::string const &id, std::shared_ptr<IObject> &&obj) {
    outputs[id] = std::move(obj);
}

}
