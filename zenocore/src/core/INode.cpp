#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/Session.h>
#include <zeno/types/DummyObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/extra/TempNode.h>
#include <zeno/utils/Error.h>
#ifdef ZENO_BENCHMARKING
#include <zeno/utils/Timer.h>
#endif
#include <zeno/utils/safe_at.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/IParam.h>
#include <zeno/DictObject.h>
#include <zeno/ListObject.h>

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
    set_output("DST", std::make_shared<DummyObject>());
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

    if (!m_dirty)
        return;

    for (const auto& param : inputs_)
        requireInput(param);

    log_debug("==> enter {}", myname);
    {
#ifdef ZENO_BENCHMARKING
        Timer _(myname);
#endif
        apply();
    }
    log_debug("==> leave {}", myname);
}

ZENO_API bool INode::requireInput(std::string const& ds) {
    return requireInput(get_input_param(ds));
}

ZENO_API bool INode::requireInput(std::shared_ptr<IParam> in_param) {
    if (!in_param)
        return false;

    if (in_param->links.empty()) {
        in_param->result = in_param->defl;
        return true;
    }

    switch (in_param->type)
    {
        case Param_Dict:
        {
            std::shared_ptr<DictObject> spDict;
            //如果只有一条边，并且对面的object刚好是一个dict，那么应该是直接连接（暂不考虑嵌套dict吧...)
            bool bDirecyLink = false;
            if (in_param->links.size() == 1)
            {
                std::shared_ptr<IParam> out_param = in_param->links[0]->fromparam.lock();
                std::shared_ptr<INode> outNode = out_param->m_spNode.lock();
                outNode->preApply();
                zany outResult = outNode->get_output(out_param->name);
                if (dynamic_cast<DictObject*>(outResult.get())) {
                    bDirecyLink = true;
                    spDict = std::dynamic_pointer_cast<DictObject>(outResult);
                }
            }
            if (!bDirecyLink)
            {
                spDict = std::make_shared<DictObject>();
                for (const auto& spLink : in_param->links)
                {
                    std::shared_ptr<IParam> outParam = in_param->links[0]->fromparam.lock();
                    std::shared_ptr<INode> outNode = outParam->m_spNode.lock();
                    outNode->preApply();
                    zany outResult = outNode->get_output(outParam->name);
                    spDict->lut[spLink->keyName] = outResult;
                }
            }
            in_param->result = spDict;   //新的写法
            break;
        }
        case Param_List:
        {
            std::shared_ptr<ListObject> spList = std::make_shared<ListObject>();
            //同上
            break;
        }
        default:
        {
            if (in_param->links.size() == 1)
            {
                std::shared_ptr<IParam> outParam = in_param->links[0]->fromparam.lock();
                std::shared_ptr<INode> outNode = outParam->m_spNode.lock();
                outNode->preApply();
                zany outResult = outNode->get_output(outParam->name);
                in_param->result = outResult;   //新的写法
            }
        }
    }
    return true;
    /*
    auto it = inputBounds.find(ds);
    if (it == inputBounds.end())
        return false;
    auto [sn, ss] = it->second;
    if (graph->applyNode(sn)) {
        auto &dc = graph->getDirtyChecker();
        dc.taintThisNode(myname);
    }
    auto ref = graph->getNodeOutput(sn, ss);
    inputs[ds] = ref;
    return true;
    */
}

ZENO_API void INode::doOnlyApply() {
    apply();
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

ZENO_API void INode::set_input_defl(std::string const& name, zany defl) {
    std::shared_ptr<IParam> param = get_input_param(name);
    param->defl = defl;
}

std::shared_ptr<IParam> INode::get_input_param(std::string const& name) const {
    for (auto& param : outputs_) {
        if (param->name == name)
            return param;
    }
    return nullptr;
}

ZENO_API bool INode::has_input(std::string const &id) const {
    return get_input_param(id) != nullptr;
    //return inputs.find(id) != inputs.end();
}

ZENO_API zany INode::get_input(std::string const &id) const {
    std::shared_ptr<IParam> param = get_input_param(id);
    return param ? param->defl : nullptr;
    /*
    if (has_keyframe(id)) {
        return get_keyframe(id);
    } else if (has_formula(id)) {
        return get_formula(id);
    }
    return safe_at(inputs, id, "input socket of node `" + myname + "`");
    */
}

ZENO_API zany INode::resolveInput(std::string const& id) {
    if (requireInput(id))
        return get_input_param(id)->result;
    else
        return nullptr;
    /*
    if (inputBounds.find(id) != inputBounds.end()) {
        if (requireInput(id))
            return get_input(id);
        else
            return nullptr;
    } else {
        auto id_ = id;
        if (inputs.find(id_) == inputs.end())
            id_.push_back(':');
        return get_input(id_);
    }
    */
}

ZENO_API void INode::set_output(std::string const &sock_name, zany obj) {
    for (auto& param : outputs_) {
        if (param->name == sock_name)
            param->result = obj;
    }
}

ZENO_API zany INode::get_output(std::string const& sock_name) {
    for (const auto& param : outputs_) {
        if (param->name == sock_name)
            return param->result;
    }
    return nullptr;
}

ZENO_API void INode::set_status(NodeStatus status)
{
    m_status = status;
    //TODO: notify mechanism
}

ZENO_API NodeStatus INode::get_status() const
{
    return m_status;
}

ZENO_API bool INode::has_keyframe(std::string const &id) const {
    return false;
    //return kframes.find(id) != kframes.end();
}

ZENO_API zany INode::get_keyframe(std::string const &id) const 
{
    std::shared_ptr<IParam> param = get_input_param(id);
    return param ? param->defl : nullptr;
    /*
    auto value = safe_at(inputs, id, "input socket of node `" + myname + "`");
    auto curves = dynamic_cast<zeno::CurveObject *>(value.get());
    if (!curves) {
        return value;
    }
    int frame = getGlobalState()->frameid;
    if (curves->keys.size() == 1) {
        auto val = curves->keys.begin()->second.eval(frame);
        value = objectFromLiterial(val);
    } else {
        int size = curves->keys.size();
        if (size == 2) {
            zeno::vec2f vec2;
            for (std::map<std::string, CurveData>::const_iterator it = curves->keys.cbegin(); it != curves->keys.cend();
                 it++) {
                int index = it->first == "x" ? 0 : 1;
                vec2[index] = it->second.eval(frame);
            }
            value = objectFromLiterial(vec2);
        } else if (size == 3) {
            zeno::vec3f vec3;
            for (std::map<std::string, CurveData>::const_iterator it = curves->keys.cbegin(); it != curves->keys.cend();
                 it++) {
                int index = it->first == "x" ? 0 : it->first == "y" ? 1 : 2;
                vec3[index] = it->second.eval(frame);
            }
            value = objectFromLiterial(vec3);
        } else if (size == 4) {
            zeno::vec4f vec4;
            for (std::map<std::string, CurveData>::const_iterator it = curves->keys.cbegin(); it != curves->keys.cend();
                 it++) {
                int index = it->first == "x" ? 0 : it->first == "y" ? 1 : it->first == "z" ? 2 : 3;
                vec4[index] = it->second.eval(frame);
            }
            value = objectFromLiterial(vec4);
        }
    }
    return value;
    */
}

ZENO_API bool INode::has_formula(std::string const &id) const {
    return false;
    //return formulas.find(id) != formulas.end();
}

ZENO_API zany INode::get_formula(std::string const &id) const 
{
    std::shared_ptr<IParam> param = get_input_param(id);
    return param ? param->defl : nullptr;
    /*
    auto value = safe_at(inputs, id, "input socket of node `" + myname + "`");
    if (auto formulas = dynamic_cast<zeno::StringObject *>(value.get())) 
    {
        std::string code = formulas->get();

        auto& desc = nodeClass->desc;
        if (!desc)
            return value;

        bool isStrFmla = false;
        for (auto const& [type, name, defl, _] : desc->inputs) {
            if (name == id && (type == "string" || type == "writepath" || type == "readpath" || type == "multiline_string")) {
                isStrFmla = true;
                break;
            }
        }
        if (!isStrFmla) {
            for (auto const& [type, name, defl, _] : desc->params) {
                auto name_ = name + ":";
                if (id == name_ &&
                    (type == "string" || type == "writepath" || type == "readpath" || type == "multiline_string")) {
                    isStrFmla = true;
                    break;
                }
            }
        }

        //remove '='
        code.replace(0, 1, "");

        if (isStrFmla) {
            auto res = getThisGraph()->callTempNode("StringEval", { {"zfxCode", objectFromLiterial(code)} }).at("result");
            value = objectFromLiterial(std::move(res));
        }
        else
        {
            std::string prefix = "vec3";
            std::string resType;
            if (code.substr(0, prefix.size()) == prefix) {
                resType = "vec3f";
            }
            else {
                resType = "float";
            }
            auto res = getThisGraph()->callTempNode("NumericEval", { {"zfxCode", objectFromLiterial(code)}, {"resType", objectFromLiterial(resType)} }).at("result");
            value = objectFromLiterial(std::move(res));
        }
    }     
    return value;
    */
}

ZENO_API TempNodeCaller INode::temp_node(std::string const &id) {
    return TempNodeCaller(graph, id);
}

//ZENO_API std::variant<int, float, std::string> INode::get_param(std::string const &id) const {
    //auto nid = id + ':';
    //if (has_input2<float>(nid)) {
        //return get_input2<float>(nid);
    //}
    //if (has_input2<int>(nid)) {
        //return get_input2<int>(nid);
    //}
    //if (has_input2<std::string>(nid)) {
        //return get_input2<std::string>(nid);
    //}
    //throw makeError("bad get_param (legacy variant mode)");
//}

}
