#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/Session.h>
#include <zeno/core/Scene.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#ifdef ZENO_VISUALIZATION  // TODO: can we decouple vis from zeno core?
#include <zeno/extra/Visualization.h>
#endif
#ifdef ZENO_GLOBALSTATE
#include <zeno/extra/GlobalState.h>
#endif
#ifdef ZENO_BENCHMARKING
#include <zeno/utils/Timer.h>
#endif
#include <zeno/utils/safe_at.h>
#include <zeno/utils/logger.h>

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
            muted_output = get_input2(desc->inputs[0].name);
        } else {
            for (auto const &[ds, bound]: inputBounds) {
                muted_output = get_input2(ds);
                break;
            }
        }
        return false;
    }

    return true;
}

ZENO_API void INode::preApply() {
    for (auto const &[ds, bound]: inputBounds) {
        requireInput(ds);
    }

    log_trace("==> enter {}", myname);
    {
#ifdef ZENO_BENCHMARKING
        Timer _(myname);
#endif
        apply();
    }
    log_trace("==> leave {}", myname);
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
    if (checkApplyCondition()) {
        log_trace("--> enter {}", myname);
        preApply();
        log_trace("--> leave {}", myname);
    }

#ifdef ZENO_VISUALIZATION
    if (has_option("VIEW")) {
        graph->hasAnyView = true;
        if (!state.isOneSubstep())  // no duplicate view when multi-substep used
            return;
        if (!graph->isViewed)  // VIEW subnodes only if subgraph is VIEW'ed
            return;
        auto desc = nodeClass->desc.get();
        auto obj = muted_output.has_value() ? muted_output
            : safe_at(outputs, desc->outputs[0].name, "output");
        if (auto p = silent_any_cast<std::shared_ptr<IObject>>(obj); p.has_value()) {
            zeno::state.addViewObject(p.value());
        }
    }
#endif
}

ZENO_API bool INode::has_option(std::string const &id) const {
    return options.find(id) != options.end();
}

ZENO_API bool INode::has_input2(std::string const &id) const {
    return inputs.find(id) != inputs.end();
}

ZENO_API zany INode::get_input2(std::string const &id) const {
    return safe_at(inputs, id, "input", myname);
}

ZENO_API void INode::set_output2(std::string const &id, zany &&obj) {
    if (auto num = silent_any_cast<std::shared_ptr<NumericObject>>(obj); num.has_value()) {
        std::visit([&] (auto const &x) {
            outputs[id] = x;
        }, num.value()->value);
    } else {
        outputs[id] = std::move(obj);
    }
}

ZENO_API std::shared_ptr<IObject> INode::get_input(std::string const &id, std::string const &msg) const {
    auto obj = get_input2(id);
    if (silent_any_cast<std::shared_ptr<IObject>>(obj).has_value())
        return safe_any_cast<std::shared_ptr<IObject>>(obj, "input `" + id + "` ");

    auto str = std::make_shared<StringObject>();
    if (auto o = exact_any_cast<std::string>(obj); o.has_value()) {
        str->set(o.value());
        return str;
    }

    auto num = std::make_shared<NumericObject>();
    using Types = typename is_variant<NumericValue>::tuple_type;
    if (static_for<0, std::tuple_size_v<Types>>([&] (auto i) {
        using T = std::tuple_element_t<i, Types>;
        if (auto o = exact_any_cast<T>(obj); o.has_value()) {
            num->set(o.value());
            return true;
        }
        return false;
    })) {
        return num;
    } else if (auto o = exact_any_cast<bool>(obj); o.has_value()) {
        num->set((int)o.value());
        return num;
    }

    throw zeno::Exception("expecting `" + msg + "` (IObject ptr) for input `"
            + id + "`, got `" + obj.type().name() + "` (any) [numeric cast also failed]");
}

ZENO_API bool INode::has_input(std::string const &id) const {
    if (!has_input2(id)) return false;
    //return inputBounds.find(id) != inputBounds.end();
    auto obj = get_input2(id);
    if (silent_any_cast<std::shared_ptr<IObject>>(obj).has_value())
        return true;

    if (exact_any_cast<std::string>(obj))
        return true;

    using Types = typename is_variant<NumericValue>::tuple_type;
    if (static_for<0, std::tuple_size_v<Types>>([&] (auto i) {
        using T = std::tuple_element_t<i, Types>;
        if (auto o = exact_any_cast<T>(obj); o.has_value()) {
            return true;
        }
        return false;
    })) {
        return true;
    } else if (auto o = exact_any_cast<bool>(obj); o.has_value()) {
        return true;
    }

    return false;
}

ZENO_API bool INode::_implicit_cast_from_to(std::string const &id,
        std::shared_ptr<IObject> const &from, std::shared_ptr<IObject> const &to) {
    auto node = graph->scene->sess->getOverloadNode("ConvertTo", {from, to});
    if (!node) {
        inputs[id] = from;
        return false;
    }
    node->doApply();
    return true;
}

}
