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
#include <filesystem>
#include <fstream>
#include <zeno/extra/GlobalComm.h>
#include <zeno/types/PrimitiveObject.h>

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

ZENO_API bool zeno::INode::getTmpCache()
{
    GlobalComm::ViewObjects objs;
    std::vector<std::string> vec;
    std::string fileName = myname + ".zenocache";
    vec.push_back(fileName);
    log_debug("get cache {}", fileName);
    bool ret = GlobalComm::fromDisk(zeno::getSession().globalComm->objTmpCachePath, zeno::getSession().globalState->frameid, objs, vec);
    if (ret)
    {
        for (const auto& [key, obj] : objs)
        {
            set_output(key, obj);
        }
    }
    return ret && objs.size() > 0;
}

ZENO_API void zeno::INode::writeTmpCaches()
{
    GlobalComm::ViewObjects objs;
    for (auto const& [name, value] : outputs) 
    {
        if (auto obj = dynamic_cast<IObject*>(value.get()))
        {
            objs.try_emplace(name, value);
        }

    }
    int frameid = zeno::getSession().globalState->frameid;
    std::vector<std::string> vec;
    std::string fileName = myname + ".zenocache";
    vec.push_back(fileName);
    GlobalComm::toDisk(zeno::getSession().globalComm->objTmpCachePath, frameid, objs, false, false, vec);
}

ZENO_API void INode::preApply() {
    auto& dc = graph->getDirtyChecker();
    if (!dc.amIDirty(myname) && bTmpCache)
    {
        if (getTmpCache())
            return;
    }
    for (auto const &[ds, bound]: inputBounds) {
        requireInput(ds);
    }

    log_debug("==> enter {}", myname);
    {
#ifdef ZENO_BENCHMARKING
        Timer _(myname);
#endif
        apply();
        if (bTmpCache)
            writeTmpCaches();
    }
    log_debug("==> leave {}", myname);
}

ZENO_API bool INode::requireInput(std::string const &ds) {
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

ZENO_API bool INode::has_input(std::string const &id) const {
    return inputs.find(id) != inputs.end();
}

ZENO_API zany INode::get_input(std::string const &id) const {
    if (has_keyframe(id)) {
        return get_keyframe(id);
    } else if (has_formula(id)) {
        return get_formula(id);
    }
    return safe_at(inputs, id, "input socket of node `" + myname + "`");
}

ZENO_API zany INode::resolveInput(std::string const& id) {
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
}

ZENO_API void INode::set_output(std::string const &id, zany obj) {
    outputs[id] = std::move(obj);
}

ZENO_API bool INode::has_keyframe(std::string const &id) const {
    return kframes.find(id) != kframes.end();
}

ZENO_API zany INode::get_keyframe(std::string const &id) const 
{
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
}

ZENO_API bool INode::has_formula(std::string const &id) const {
    return formulas.find(id) != formulas.end();
}

ZENO_API zany INode::get_formula(std::string const &id) const 
{
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
