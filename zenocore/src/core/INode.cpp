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
#include <zeno/utils/string.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#ifdef ZENO_BENCHMARKING
#include <zeno/utils/Timer.h>
#endif
#include <zeno/utils/safe_at.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/IParam.h>
#include <zeno/DictObject.h>
#include <zeno/ListObject.h>
#include <zeno/utils/helper.h>

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

ZENO_API zvariant INode::get_input_defl(std::string const& name)
{
    std::shared_ptr<IParam> param = get_input_param(name);
    return param->defl;
}

ZENO_API std::string INode::get_nodecls() const
{
    return nodecls;
}

ZENO_API std::string INode::get_ident() const
{
    return name;
}

ZENO_API std::string INode::get_name() const
{
    return name;
}

ZENO_API void INode::set_name(std::string const& customname)
{
    name = customname;
}

ZENO_API void INode::set_view(bool bOn)
{
    if (bOn)
        m_status = m_status | NodeStatus::View;
    else
        m_status = m_status ^ NodeStatus::View;
}

ZENO_API bool INode::is_view() const
{
    return m_status & NodeStatus::View;
}

ZENO_API void INode::mark_dirty(bool bOn)
{
    m_dirty = bOn;
}

ZENO_API bool INode::is_dirty() const
{
    return m_dirty;
}

ZENO_API void INode::complete() {}

ZENO_API void INode::preApply() {

    if (!m_dirty)
        return;

    for (const auto& param : inputs_) {
        bool ret = requireInput(param);
        if (ret)
            zeno::log_warn("the param {} may not be initialized", param);
    }


    log_debug("==> enter {}", name);
    {
#ifdef ZENO_BENCHMARKING
        Timer _(name);
#endif
        apply();
    }
    log_debug("==> leave {}", name);
}

ZENO_API bool INode::requireInput(std::string const& ds) {
    return requireInput(get_input_param(ds));
}

ZENO_API bool INode::requireInput(std::shared_ptr<IParam> in_param) {
    if (!in_param)
        return false;

    if (in_param->links.empty()) {
        in_param->result = process(in_param);
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
                outNode->doApply();
                zany outResult = outNode->get_output(out_param->name);
                if (spDict = std::dynamic_pointer_cast<DictObject>(outResult)) {
                    bDirecyLink = true;
                }
            }
            if (!bDirecyLink)
            {
                spDict = std::make_shared<DictObject>();
                for (const auto& spLink : in_param->links)
                {
                    const std::string& keyName = spLink->keyName;
                    std::shared_ptr<IParam> outParam = in_param->links[0]->fromparam.lock();
                    std::shared_ptr<INode> outNode = outParam->m_spNode.lock();
                    outNode->doApply();
                    zany outResult = outNode->get_output(outParam->name);
                    spDict->lut[keyName] = outResult;
                }
            }
            in_param->result = spDict;
            break;
        }
        case Param_List:
        {
            std::shared_ptr<ListObject> spList;
            bool bDirectLink = false;
            if (in_param->links.size() == 1)
            {
                std::shared_ptr<IParam> out_param = in_param->links[0]->fromparam.lock();
                std::shared_ptr<INode> outNode = out_param->m_spNode.lock();
                outNode->doApply();
                zany outResult = outNode->get_output(out_param->name);
                if (spList = std::dynamic_pointer_cast<ListObject>(outResult))
                    bDirectLink = true;
            }
            if (!bDirectLink)
            {
                spList = std::make_shared<ListObject>();
                for (const auto& spLink : in_param->links)
                {
                    //list的情况下，keyName是不是没意义，顺序怎么维持？
                    std::shared_ptr<IParam> outParam = in_param->links[0]->fromparam.lock();
                    std::shared_ptr<INode> outNode = outParam->m_spNode.lock();
                    outNode->doApply();
                    zany outResult = outNode->get_output(outParam->name);
                    spList->arr.push_back(outResult);
                }
            }
            in_param->result = spList;
            //同上
            break;
        }
        default:
        {
            if (in_param->links.size() == 1)
            {
                std::shared_ptr<IParam> outParam = in_param->links[0]->fromparam.lock();
                std::shared_ptr<INode> outNode = outParam->m_spNode.lock();
                outNode->doApply();
                zany outResult = outNode->get_output(outParam->name);
                in_param->result = outResult;
            }
        }
    }
    return true;
}

ZENO_API void INode::doOnlyApply() {
    apply();
}

ZENO_API void INode::doApply() {
    //if (checkApplyCondition()) {
    log_trace("--> enter {}", name);
    preApply();
    log_trace("--> leave {}", name);
}

ZENO_API std::vector<std::shared_ptr<IParam>> INode::get_input_params() const
{
    return inputs_;
}

ZENO_API std::vector<std::shared_ptr<IParam>> INode::get_output_params() const
{
    return outputs_;
}

ZENO_API void INode::set_input_defl(std::string const& name, zvariant defl) {
    std::shared_ptr<IParam> param = get_input_param(name);
    param->defl = defl;
}

ZENO_API std::shared_ptr<IParam> INode::get_input_param(std::string const& name) const {
    for (auto& param : inputs_) {
        if (param->name == name)
            return param;
    }
    return nullptr;
}

void INode::add_input_param(std::shared_ptr<IParam> param) {
    inputs_.push_back(param);
}

void INode::add_output_param(std::shared_ptr<IParam> param) {
    outputs_.push_back(param);
}

ZENO_API std::shared_ptr<IParam> INode::get_output_param(std::string const& name) const {
    for (auto& param : outputs_) {
        if (param->name == name)
            return param;
    }
    return nullptr;
}

ZENO_API bool INode::update_param(const std::string& name, const zvariant& new_value) {
    for (auto& param : inputs_) {
        if (param->name == name)
        {
            if (!zeno::isEqual(param->defl, new_value, param->type))
            {
                zvariant old_value = param->defl;
                param->defl = new_value;
                CALLBACK_NOTIFY(update_param, name, old_value, new_value)
                return true;
            }
        }
    }
    return false;
}

void INode::directly_setinputs(std::map<std::string, zany> inputs)
{
    for (auto& [name, val] : inputs) {
        std::shared_ptr<IParam> sparam = get_input_param(name);
        if (!sparam) {
            sparam = std::make_shared<IParam>();
            sparam->name = name;
            sparam->m_spNode = shared_from_this();
            sparam->type = Param_Null;
            sparam->defl = zvariant();
        }
        sparam->result = val;
    }
}

std::map<std::string, zany> INode::getoutputs()
{
    std::map<std::string, zany> outputs;
    for (auto param : outputs_) {
        outputs.insert(std::make_pair(param->name, param->result));
    }
    return outputs;
}

std::vector<std::pair<std::string, zany>> INode::getinputs()
{
    std::vector<std::pair<std::string, zany>> inputs;
    for (auto param : inputs_) {
        inputs.push_back(std::make_pair(param->name, param->result));
    }
    return inputs;
}

std::pair<std::string, std::string> INode::getinputbound(std::string const& name, std::string const& msg) const
{
    for (auto param : inputs_) {
        if (param->name == name && !param->links.empty()) {
            auto lnk = param->links[0];
            auto outparam = lnk->fromparam.lock();
            if (outparam) {
                outparam->name;
                auto pnode = outparam->m_spNode.lock();
                if (pnode) {
                    auto id = pnode->get_ident();
                    return { id, outparam->name };
                }
            }
        }
    }
    throw makeError<KeyError>(name, msg);
}

std::vector<std::pair<std::string, zany>> INode::getoutputs2()
{
    std::vector<std::pair<std::string, zany>> outputs;
    for (auto param : outputs_) {
        outputs.push_back(std::make_pair(param->name, param->result));
    }
    return outputs;
}

void INode::init(const NodeData& dat)
{
    if (dat.name.empty())
        name = dat.name;
    for (const ParamInfo& param : dat.inputs)
    {
        std::shared_ptr<IParam> sparam = get_input_param(param.name);
        if (!sparam) {
            zeno::log_warn("input param `{}` is not registerd in current zeno version");
            continue;
        }
        sparam->defl = param.defl;
        sparam->name = param.name;
        sparam->type = param.type;
        sparam->m_spNode = shared_from_this();
    }
    for (const ParamInfo& param : dat.outputs)
    {
        std::shared_ptr<IParam> sparam = get_output_param(param.name);
        if (!sparam) {
            zeno::log_warn("output param `{}` is not registerd in current zeno version");
            continue;
        }
        sparam->defl = param.defl;
        sparam->name = name;
        sparam->type = param.type;
        sparam->m_spNode = shared_from_this();
    }
}

ZENO_API bool INode::has_input(std::string const &id) const {
    return get_input_param(id) != nullptr;
    //return inputs.find(id) != inputs.end();
}

ZENO_API zany INode::get_input(std::string const &id) const {
    std::shared_ptr<IParam> param = get_input_param(id);
    return param ? param->result : nullptr;
}

ZENO_API zany INode::resolveInput(std::string const& id) {
    if (requireInput(id))
        return get_input_param(id)->result;
    else
        return nullptr;
}

ZENO_API bool INode::set_input(std::string const& name, zany obj) {
    for (auto& param : inputs_) {
        if (param->name == name) {
            param->result = obj;
            return true;
        }
    }
    return false;
}

ZENO_API bool INode::has_output(std::string const& name) const {
    return get_output_param(name) != nullptr;
}

ZENO_API bool INode::set_output(std::string const &sock_name, zany obj) {
    for (auto& param : outputs_) {
        if (param->name == sock_name) {
            param->result = obj;
            return true;
        }
    }
    return false;
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
    //deprecated: will parse it when processing defl value
    return nullptr;
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
    //deprecated: will parse it when processing defl value
    return nullptr;
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

float INode::resolve(const std::string& formulaOrKFrame, const ParamType type)
{
    int frame = getGlobalState()->frameid;
    if (zeno::starts_with(formulaOrKFrame, "=")) {
        std::string code = formulaOrKFrame.substr(1);
        auto res = getThisGraph()->callTempNode(
            "NumericEval", { {"zfxCode", objectFromLiterial(code)}, {"resType", objectFromLiterial("float")} }).at("result");
        assert(res);
        std::shared_ptr<zeno::NumericObject> num = std::dynamic_pointer_cast<zeno::NumericObject>(res);
        float fVal = num->get<float>();
        return fVal;
    }
    else if (zany curve = zeno::parseCurveObj(formulaOrKFrame)) {
        std::shared_ptr<zeno::CurveObject> curves = std::dynamic_pointer_cast<zeno::CurveObject>(curve);
        assert(curves && curves->keys.size() == 1);
        float fVal = curves->keys.begin()->second.eval(frame);
        return fVal;
    }
    else {
        if (Param_Float == type)
        {
            float fVal = std::stof(formulaOrKFrame);
            return fVal;
        }
        else {
            float fVal = std::stoi(formulaOrKFrame);
            return fVal;
        }
    }
}

template<class T, class E> zany INode::resolveVec(const zvariant& defl, const ParamType type)
{
    if (std::holds_alternative<T>(defl)) {
        return std::make_shared<zeno::NumericObject>(std::get<T>(defl));
    }
    else if (std::holds_alternative<E>(defl)) {
        E vec = std::get<E>(defl);
        T vecnum;
        for (int i = 0; i < vec.size(); i++) {
            float fVal = resolve(vec[i], type);
            vecnum[i] = fVal;
        }
        return std::make_shared<zeno::NumericObject>(vecnum);
    }
    else {
        //error, throw expection.
        return nullptr;
    }
}

zany INode::process(std::shared_ptr<IParam> in_param)
{
    if (!in_param) {
        return nullptr;
    }

    int frame = getGlobalState()->frameid;
    zany result;

    const ParamType type = in_param->type;
    const zvariant defl = in_param->defl;

    switch (type) {
        case Param_Int:
        case Param_Float:
        case Param_Bool:
        {
            //先不考虑int float的划分,直接按variant的值来。
            zvariant resolve_value;
            if (std::holds_alternative<std::string>(defl))
            {
                std::string str = std::get<std::string>(defl);
                float fVal = resolve(str, type);
                result = std::make_shared<zeno::NumericObject>(fVal);
            }
            else if (std::holds_alternative<int>(defl))
            {
                result = std::make_shared<zeno::NumericObject>(std::get<int>(defl));
            }
            else if (std::holds_alternative<float>(defl))
            {
                result = std::make_shared<zeno::NumericObject>(std::get<float>(defl));
            }
            else
            {
                //error, throw expection.
            }
            break;
        }
        case Param_String:
        {
            if (std::holds_alternative<std::string>(defl))
            {
                std::string str = std::get<std::string>(defl);
                result = std::make_shared<zeno::StringObject>(str);
            }
            else {
                //error, throw expection.
            }
            break;
        }
        case Param_Vec2f:   result = resolveVec<vec2f, vec2s>(defl, type);  break;
        case Param_Vec2i:   result = resolveVec<vec2i, vec2s>(defl, type);  break;
        case Param_Vec3f:   result = resolveVec<vec3f, vec3s>(defl, type);  break;
        case Param_Vec3i:   result = resolveVec<vec3i, vec3s>(defl, type);  break;
        case Param_Vec4f:   result = resolveVec<vec4f, vec4s>(defl, type);  break;
        case Param_Vec4i:   result = resolveVec<vec4i, vec4s>(defl, type);  break;
        case Param_Curve:
            break;  //TODO
        case Param_List:
        {
            //TODO: List现在还没有ui支持，而且List是泛型容器，对于非Literal值不好设定默认值。
            break;
        }
        case Param_Dict:
        {
            break;
        }
    }
    return result;
}

}
