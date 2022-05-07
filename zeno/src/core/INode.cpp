#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/Session.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/Error.h>
#ifdef ZENO_BENCHMARKING
#include <zeno/utils/Timer.h>
#endif
#include <zeno/utils/safe_at.h>
#include <zeno/utils/logger.h>

namespace zeno {

ZENO_API INode::INode() = default;
ZENO_API INode::~INode() = default;

ZENO_API Graph *INode::getThisGraph() const {
    return graph;
}

ZENO_API Session *INode::getThisSession() const {
    return graph->session;
}

ZENO_API GlobalState *INode::getGlobalState() const {
    return graph->session->globalState.get();
}

ZENO_API void INode::doComplete() {
    set_output("DST", std::make_shared<ConditionObject>());
    complete();
}

ZENO_API void INode::complete() {}

/*ZENO_API bool INode::checkApplyCondition() {
    if (has_option("ONCE")) {  // TODO: frame control should be editor work
        if (!getGlobalState()->isFirstSubstep())
            return false;
    }

    if (has_option("PREP")) {
        if (!getGlobalState()->isOneSubstep())
            return false;
    }

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
}*/

ZENO_API void INode::preApply() {
    for (auto const &[ds, bound]: inputBounds) {
        requireInput(ds);
    }

    log_debug("==> enter {}", myname);
    {
#ifdef ZENO_BENCHMARKING
        Timer _(myname);
#endif
        apply();
    }
    log_debug("==> leave {}", myname);
}

ZENO_API bool INode::requireInput(std::string const &ds) {
    auto it = inputBounds.find(ds);
    if (it == inputBounds.end())
        return false;
    auto [sn, ss] = it->second;
    graph->applyNode(sn);
    auto ref = graph->getNodeOutput(sn, ss);
    inputs[ds] = ref;
    return true;
}

ZENO_API void INode::doApply() {
    //if (checkApplyCondition()) {
    log_trace("--> enter {}", myname);
    preApply();
    log_trace("--> leave {}", myname);
    //}

    /*if (has_option("VIEW")) {
        graph->hasAnyView = true;
        if (!getGlobalState()->isOneSubstep())  // no duplicate view when multi-substep used
            return;
        if (!graph->isViewed)  // VIEW subnodes only if subgraph is VIEW'ed
            return;
        auto desc = nodeClass->desc.get();
        auto obj = muted_output ? muted_output
            : safe_at(outputs, desc->outputs[0].name, "output");
        if (auto p = std::dynamic_pointer_cast<IObject>(obj); p) {
            getGlobalState()->addViewObject(p);
        }
    }*/
}

/*ZENO_API bool INode::has_option(std::string const &id) const {
    return options.find(id) != options.end();
}*/

ZENO_API bool INode::has_input(std::string const &id) const {
    return inputs.find(id) != inputs.end();
}

ZENO_API zany INode::get_input(std::string const &id) const {
    return safe_at(inputs, id, "input socket name of node " + myname);
}

ZENO_API void INode::set_output(std::string const &id, zany obj) {
    outputs[id] = std::move(obj);
}

ZENO_API std::variant<int, float, std::string> INode::get_param(std::string const &id) const {
    auto nid = id + ':';
    if (has_input2<int>(nid)) {
        return get_input2<int>(nid);
    }
    if (has_input2<float>(nid)) {
        return get_input2<float>(nid);
    }
    if (has_input2<std::string>(nid)) {
        return get_input2<std::string>(nid);
    }
    throw makeError("bad get_param (variant mode)");
}

}
