#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/Session.h>
#include <zeno/core/Assets.h>
#include <zeno/core/ObjectManager.h>
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
#include <zeno/utils/uuid.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/CoreParam.h>
#include <zeno/DictObject.h>
#include <zeno/ListObject.h>
#include <zeno/utils/helper.h>
#include <zeno/utils/uuid.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/GraphException.h>
#include <zeno/formula/formula.h>
#include <zeno/core/ReferManager.h>
#include <zeno/core/FunctionManager.h>
#include "reflect/type.hpp"
#include <zeno/types/MeshObject.h>
#include "zeno_types/reflect/reflection.generated.hpp"


using namespace zeno::reflect;
using namespace zeno::types;


namespace zeno {

ZENO_API INode::INode() {}

void INode::initUuid(std::shared_ptr<Graph> pGraph, const std::string nodecls) {
    m_nodecls = nodecls;
    this->graph = pGraph;

    m_uuid = generateUUID(nodecls);
    ObjPath path;
    path += m_uuid;
    while (pGraph) {
        const std::string name = pGraph->getName();
        if (name == "main") {
            break;
        }
        else {
            if (!pGraph->optParentSubgNode.has_value())
                break;
            auto pSubnetNode = pGraph->optParentSubgNode.value();
            assert(pSubnetNode);
            path = (pSubnetNode->m_uuid) + "/" + path;
            pGraph = pSubnetNode->graph.lock();
        }
    }
    m_uuidPath = path;
}

ZENO_API INode::~INode() = default;

ZENO_API std::shared_ptr<Graph> INode::getThisGraph() const {
    return graph.lock();
}

ZENO_API Session *INode::getThisSession() const {
    return &getSession();
}

ZENO_API GlobalState *INode::getGlobalState() const {
    return getSession().globalState.get();
}

ZENO_API void INode::doComplete() {
    set_output("DST", std::make_shared<DummyObject>());
    complete();
}

ZENO_API std::string INode::get_nodecls() const
{
    return m_nodecls;
}

ZENO_API std::string INode::get_ident() const
{
    return m_name;
}

ZENO_API std::string INode::get_show_name() const {
    if (nodeClass) {
        std::string dispName = nodeClass->m_customui.nickname;
        if (!dispName.empty())
            return dispName;
    }
    return m_nodecls;
}

ZENO_API std::string INode::get_show_icon() const {
    if (nodeClass) {
        return nodeClass->m_customui.iconResPath;
    }
    else {
        return "";
    }
}

ZENO_API CustomUI INode::get_customui() const {
    if (nodeClass) {
        return nodeClass->m_customui;
    }
    else {
        return CustomUI();
    }
}

ZENO_API ObjPath INode::get_graph_path() const {
    ObjPath path;
    path = "";

    std::shared_ptr<Graph> pGraph = graph.lock();

    while (pGraph) {
        const std::string name = pGraph->getName();
        if (name == "main") {
            path = "/main/" + path;
            break;
        }
        else {
            if (!pGraph->optParentSubgNode.has_value())
                break;
            auto pSubnetNode = pGraph->optParentSubgNode.value();
            assert(pSubnetNode);
            path = pSubnetNode->m_name + "/" + path;
            pGraph = pSubnetNode->graph.lock();
        }
    }
    return path;
}

ZENO_API CustomUI INode::export_customui() const
{
    std::set<std::string> intputPrims, outputPrims, inputObjs, outputObjs;
    zeno::CustomUI origin = nodeClass->m_customui;
    zeno::CustomUI exportui;
    exportui.nickname = origin.nickname;
    exportui.iconResPath = origin.iconResPath;
    exportui.doc = origin.doc;
    if (!origin.category.empty())
        exportui.category = origin.category;

    zeno::ParamGroup exportgroup;
    zeno::ParamTab exporttab;
    if (!origin.inputPrims.tabs.empty()) {
        exporttab.name = origin.inputPrims.tabs[0].name;
        if (!origin.inputPrims.tabs[0].groups.empty()) {
            exportgroup.name = origin.inputPrims.tabs[0].groups[0].name;
        }
    }
    for (const zeno::ParamTab& tab : origin.inputPrims.tabs) {
        for (const zeno::ParamGroup& group : tab.groups) {
            for (const zeno::ParamPrimitive& param : group.params) {
                auto iter = m_inputPrims.find(param.name);
                if (iter != m_inputPrims.end()) {
                    exportgroup.params.push_back(iter->second.exportParam());
                    intputPrims.insert(param.name);
                }
            }
        }
    }
    for (const zeno::ParamPrimitive& param : origin.outputPrims) {
        auto iter = m_outputPrims.find(param.name);
        if (iter != m_outputPrims.end()) {
            exportui.outputPrims.push_back(iter->second.exportParam());
            outputPrims.insert(param.name);
        }
    }
    for (const auto& param : origin.inputObjs) {
        auto iter = m_inputObjs.find(param.name);
        if (iter != m_inputObjs.end()) {
            exportui.inputObjs.push_back(iter->second.exportParam());
            inputObjs.insert(param.name);
        }
    }
    for (const auto& param : origin.outputObjs) {
        auto iter = m_outputObjs.find(param.name);
        if (iter != m_outputObjs.end()) {
            exportui.outputObjs.push_back(iter->second.exportParam());
            outputObjs.insert(param.name);
        }
    }
    exporttab.groups.emplace_back(std::move(exportgroup));
    exportui.inputPrims.tabs.emplace_back(std::move(exporttab));
    for (auto& [key, param] : m_inputPrims) {
        if (intputPrims.find(key) == intputPrims.end())
            exportui.inputPrims.tabs[0].groups[0].params.push_back(param.exportParam());
    }
    for (auto& [key, param] : m_outputPrims) {
        if (outputPrims.find(key) == outputPrims.end())
            exportui.outputPrims.push_back(param.exportParam());
    }
    for (auto& [key, param] : m_inputObjs) {
        if (inputObjs.find(key) == inputObjs.end())
            exportui.inputObjs.push_back(param.exportParam());
    }
    for (auto& [key, param] : m_outputObjs) {
        if (outputObjs.find(key) == outputObjs.end())
            exportui.outputObjs.push_back(param.exportParam());
    }
    return exportui;
}

ZENO_API ObjPath INode::get_path() const {
    ObjPath path;
    path = m_name;

    std::shared_ptr<Graph> pGraph = graph.lock();

    while (pGraph) {
        const std::string name = pGraph->getName();
        if (name == "main") {
            path = "/main/" + path;
            break;
        }
        else {
            if (!pGraph->optParentSubgNode.has_value())
                break;
            auto pSubnetNode = pGraph->optParentSubgNode.value();
            assert(pSubnetNode);
            path = pSubnetNode->m_name + "/" + path;
            pGraph = pSubnetNode->graph.lock();
        }
    }
    return path;
}

std::string INode::get_uuid() const
{
    return m_uuid;
}

ZENO_API std::string INode::get_name() const
{
    return m_name;
}

ZENO_API void INode::set_name(const std::string& customname)
{
    m_name = customname;
}

ZENO_API void INode::set_view(bool bOn)
{
    CORE_API_BATCH

    m_bView = bOn;
    CALLBACK_NOTIFY(set_view, m_bView)

    std::shared_ptr<Graph> spGraph = graph.lock();
    assert(spGraph);
    spGraph->viewNodeUpdated(m_name, bOn);
}

ZENO_API bool INode::is_view() const
{
    return m_bView;
}

void INode::reportStatus(bool bDirty, NodeRunStatus status) {
    m_status = status;
    m_dirty = bDirty;
    zeno::getSession().reportNodeStatus(m_uuidPath, bDirty, status);
}

void INode::mark_previous_ref_dirty() {
    mark_dirty(true);
    //不仅要自身标脏，如果前面的节点是以引用的方式连接，说明前面的节点都可能被污染了，所有都要标脏。
    //TODO: 由端口而不是边控制。
    /*
    for (const auto& [name, param] : m_inputs) {
        for (const auto& link : param.links) {
            if (link->lnkProp == Link_Ref) {
                auto spOutParam = link->fromparam.lock();
                auto spPreviusNode = spOutParam->m_wpNode.lock();
                spPreviusNode->mark_previous_ref_dirty();
            }
        }
    }
    */
}

void INode::onInterrupted() {
    mark_dirty(true);
    mark_previous_ref_dirty();
}

ZENO_API void INode::mark_dirty(bool bOn, bool bWholeSubnet, bool bRecursively)
{
    scope_exit sp([&] {
        m_status = Node_DirtyReadyToRun;  //修改了数据，标脏，并置为此状态。（后续在计算过程中不允许修改数据，所以markDirty理论上是前端驱动）
        reportStatus(m_dirty, m_status);
    });

    if (m_dirty == bOn)
        return;

    m_dirty = bOn;

    if (!bRecursively)
        return;

    if (m_dirty) {
        for (auto& [name, param] : m_inputObjs) {   //如果input对象是owning类型，mark_dirty时需要将上游也mark_dirty
            if (param.socketType == Socket_Owning) {
                for (auto link : param.links) {
                    auto fromParam = link->fromparam;
                    assert(fromParam);
                    if (fromParam) {
                        auto fromNode = fromParam->m_wpNode.lock();
                        assert(fromNode);
                        fromNode->mark_dirty(m_dirty);
                    }
                }
            }
        }
        for (auto& [name, param] : m_outputObjs) {
            for (auto link : param.links) {
                auto inParam = link->toparam;
                assert(inParam);
                if (inParam) {
                    auto inNode = inParam->m_wpNode.lock();
                    assert(inNode);
                    inNode->mark_dirty(m_dirty);
                }
            }
        }
        for (auto& [name, param] : m_outputPrims) {
            for (auto link : param.links) {
                auto inParam = link->toparam;
                assert(inParam);
                if (inParam) {
                    auto inNode = inParam->m_wpNode.lock();
                    assert(inNode);
                    inNode->mark_dirty(m_dirty);
                }
            }
        }
    }

    if (SubnetNode* pSubnetNode = dynamic_cast<SubnetNode*>(this))
    {
        if (bWholeSubnet)
            pSubnetNode->mark_subnetdirty(bOn);
    }

    std::shared_ptr<Graph> spGraph = graph.lock();
    assert(spGraph);
    if (spGraph->optParentSubgNode.has_value())
    {
        spGraph->optParentSubgNode.value()->mark_dirty(true, false);
    }
}

void INode::mark_dirty_objs()
{
    for (auto const& [name, param] : m_outputObjs)
    {
        if (param.spObject) {
            assert(param.spObject);
            if (param.spObject->key().empty()) {
                continue;
            }
            getSession().objsMan->collect_removing_objs(param.spObject->key());
        }
    }
}

ZENO_API void INode::complete() {}

ZENO_API void INode::preApply() {
    if (!m_dirty)
        return;

    reportStatus(true, Node_Pending);

    //TODO: the param order should be arranged by the descriptors.
    for (const auto& [name, param] : m_inputObjs) {
        bool ret = requireInput(name);
        if (!ret)
            zeno::log_warn("the param {} may not be initialized", name);
    }
    for (const auto& [name, param] : m_inputPrims) {
        bool ret = requireInput(name);
        if (!ret)
            zeno::log_warn("the param {} may not be initialized", name);
    }
}

ZENO_API void INode::apply() {

}

ZENO_API void INode::reflecNode_apply()
{
    if (m_pTypebase) {
        for (zeno::reflect::IMemberFunction* func : m_pTypebase->get_member_functions()) {
            const auto& funcname = func->get_name();
            if (funcname == "apply") {
                //根据ReflectCustomUI获取fieldName到displayName映射
                std::map<std::string, std::string> inputPrims, outputPrims, inputObjs, outputObjs;
                getFieldNameParamNameMapByReflectCustomUi(m_pTypebase, shared_from_this(), inputPrims, outputPrims, inputObjs, outputObjs);
                const auto& getOutputParamNameFromFieldName = [&outputPrims, &outputObjs](std::string paramname, bool isPrim) -> std::string {
                    if (!isPrim) {
                        return outputObjs.find(paramname) == outputObjs.end() ? paramname : outputObjs[paramname];
                    }
                    else {
                        return outputPrims.find(paramname) == outputPrims.end() ? paramname : outputPrims[paramname];
                    }
                };
                //从apply参数获取输入
                zeno::reflect::ArrayList<zeno::reflect::Any> paramValues;
                std::vector<std::tuple<std::string, zeno::ParamType, int>> outputsName;

                const zeno::reflect::ArrayList<zeno::reflect::RTTITypeInfo>& params = func->get_params();
                const auto& field_names = func->get_params_name();
                for (int i = 0; i < params.size(); i++) {
                    const zeno::reflect::RTTITypeInfo& param_type = params[i];
                    if (!param_type.has_flags(zeno::reflect::TF_IsConst) && param_type.has_flags(zeno::reflect::TF_IsLValueRef)) {
                        ParamType _type = param_type.get_decayed_hash() == 0 ? param_type.hash_code() : param_type.get_decayed_hash();
                        bool bConstPtr;
                        outputsName.push_back({ getOutputParamNameFromFieldName(field_names[i].c_str(), zeno::isObjectType(param_type, bConstPtr)), _type, i});
                    }
                    else {
                        zeno::reflect::Any inputAny;
                        bool bConstPtr;
                        std::string inputName;
                        if (zeno::isObjectType(param_type, bConstPtr)) {
                            inputName = inputObjs.find(field_names[i].c_str()) == inputObjs.end() ? field_names[i].c_str() : inputObjs[field_names[i].c_str()];
                        }
                        else {
                            inputName = inputPrims.find(field_names[i].c_str()) == inputPrims.end() ? field_names[i].c_str() : inputPrims[field_names[i].c_str()];
                        }
                        auto iter = m_inputPrims.find(inputName);
                        if (iter != m_inputPrims.end()) {
                            auto& val = iter->second.result;
                            if (val.has_value()) {
                                inputAny = val;
                            }
                            else {
                                inputAny = iter->second.defl;
                            }
                        }
                        else {
                            auto iter2 = m_inputObjs.find(inputName);
                            if (iter2 != m_inputObjs.end()) {
                                inputAny = iter2->second.spObject;
                        }
                        }
                        paramValues.add_item(inputAny);
                    }
                }
                for (auto& paramInfo : outputsName) {
                    ParamType type = std::get<1>(paramInfo);
                    int idx = std::get<2>(paramInfo);
                    //TODO: 这里只能初始化ctor没有参数的类型
                    paramValues.add_item(func->init_param_default_value(idx));
                }
                //从输入到成员变量
                for (zeno::reflect::IMemberField* field : m_pTypebase->get_member_fields()) {
                    std::string field_name(field->get_name().c_str());
                    std::string param_name;
                    if (const zeno::reflect::IRawMetadata* metadata = field->get_metadata()) {
                    if (const zeno::reflect::IMetadataValue* value = metadata->get_value("Role")) {
                            int _role = value->as_int();
                            if (_role == Role_InputPrimitive || _role == Role_InputObject) {
                                if (const zeno::reflect::IMetadataValue* value = metadata->get_value("DisplayName"))
                                    param_name = value->as_string();
                                else {
                                    param_name = field_name;
                                }
                                zeno::reflect::Any inputAny;
                                auto iter = m_inputPrims.find(param_name);
                                if (iter != m_inputPrims.end()) {
                                    auto& val = iter->second.result;
                                    if (val.has_value()) {
                                        inputAny = val;
                                    }
                                    else {
                                        inputAny = iter->second.defl;
                                    }
                                }
                                else {
                                    auto iter2 = m_inputObjs.find(param_name);
                                    if (iter2 != m_inputObjs.end()) {
                                        inputAny = iter2->second.spObject;
                                }
                                }
                                if (inputAny.has_value())
                                    field->set_field_value(this, inputAny);
                            }
                        }
                    }
                }

                //调用apply
                zeno::reflect::Any res = func->invoke_unsafe(this, paramValues);

                //从apply参数/返回值获取输出
                for (int i = paramValues.size() - 1; i > paramValues.size() - outputsName.size() - 1; i--) {
                    auto iter = m_outputObjs.find(getOutputParamNameFromFieldName(field_names[i].c_str(), false));
                    if (iter != m_outputObjs.end()) {
                        if (paramValues[i].has_value() && params[i].has_flags(TF_IsObject)) {
                            //TODO: 从引用参数传出至shared_ptr<IObject>
                            //iter->second.spObject = zeno::reflect::any_cast<std::shared_ptr<zeno::IObject>>(paramValues[i]);
                        }
                    }
                    else {
                        auto iter2 = m_outputPrims.find(getOutputParamNameFromFieldName(field_names[i].c_str(), true));
                        if (iter2 != m_outputPrims.end()) {
                            iter2->second.result = zeno::reflect::move(paramValues[i]);
                        }
                    }
                }
                const zeno::reflect::RTTITypeInfo& rtti = func->get_return_rtti();
                ParamType _type = rtti.get_decayed_hash() == 0 ? rtti.hash_code() : rtti.get_decayed_hash();
                bool bConstPtr = false;
                if (zeno::isObjectType(rtti, bConstPtr)) {
                    //apply函数的返回值没法用命名表达，所以应该是空字符串
                    auto iter = outputObjs.find("");
                    if (iter != outputObjs.end()) {
                        auto _iter = m_outputObjs.find(iter->second);
                        if (_iter != m_outputObjs.end()) {
                            _iter->second.spObject = any_cast<zany>(res);
                        }
                    }
                    else {
                        zeno::log_warn("invalid output object name");
                    }
                }
                else {
                    auto iter = outputPrims.find("");
                    if (iter != outputPrims.end()) {
                        auto _iter = m_outputPrims.find(iter->second);
                        if (_iter != m_outputPrims.end()) {
                            _iter->second.result = res;
                        }
                    }
                    else {
                        zeno::log_warn("invalid output prim name");
                    }
                }
                //从成员变量到输入
                for (zeno::reflect::IMemberField* field : m_pTypebase->get_member_fields()) {
                    if (const zeno::reflect::IRawMetadata* metadata = field->get_metadata()) {
                        if (const zeno::reflect::IMetadataValue* value = metadata->get_value("Role")) {
                            int _role = value->as_int();
                            if (_role == Role_OutputPrimitive || _role == Role_OutputObject) {
                                std::string field_name(field->get_name().c_str());
                                std::string param_name;
                                if (const zeno::reflect::IMetadataValue* value = metadata->get_value("DisplayName"))
                                    param_name = value->as_string();
                                else {
                                    param_name = field_name;
                                }
                                zeno::reflect::Any outputAny = field->get_field_value(this);
                                if (outputAny.has_value()) {
                                    auto iter = m_outputPrims.find(param_name);
                                    if (iter != m_outputPrims.end()) {
                                        iter->second.result = zeno::reflect::move(outputAny);
                                    }
                                    else {
                                        auto iter2 = m_outputObjs.find(param_name);
                                        if (iter2 != m_outputObjs.end())
                                        {
                                            //TODO: need to parse on the param, not only return value.
                                            //iter2->second.spObject = zeno::reflect::any_cast<std::shared_ptr<IObject>>(outputAny);
                                    }
                                }
                            }
                        }
                    }
                }
                }
                break;
            }
        }
    }
}

ZENO_API void INode::registerObjToManager()
{
    for (auto const& [name, param] : m_outputObjs)
    {
        if (param.spObject)
        {
            if (std::dynamic_pointer_cast<NumericObject>(param.spObject) ||
                std::dynamic_pointer_cast<StringObject>(param.spObject)) {
                return;
            }

            if (param.spObject->key().empty())
            {
                //如果当前节点是引用前继节点产生的obj，则obj.key不为空，此时就必须沿用之前的id，
                //以表示“引用”，否则如果新建id，obj指针可能是同一个，会在manager引起混乱。
                param.spObject->update_key(m_uuid);
            }

            const std::string& key = param.spObject->key();
            assert(!key.empty());
            param.spObject->nodeId = m_name;

            auto& objsMan = getSession().objsMan;
            std::shared_ptr<INode> spNode = shared_from_this();
            objsMan->collectingObject(param.spObject, spNode, m_bView);
        }
    }
}

std::set<std::pair<std::string, std::string>> INode::resolveReferSource(std::string const& primitive_param) {

    std::set<std::pair<std::string, std::string>> refSources;

    auto iter = m_inputPrims.find(primitive_param);
    if (iter == m_inputPrims.end())
        return refSources;

    std::vector<std::string> refSegments;
    const Any& param_defl = iter->second.defl;

    ParamType deflType = param_defl.type().hash_code();
    if (deflType == zeno::types::gParamType_String) {
        const std::string& param_text = zeno::reflect::any_cast<std::string>(param_defl);
        if (param_text.find("ref(") == std::string::npos) {
            return refSources;
        }
        refSegments.push_back(param_text);
    }
    else if (deflType == zeno::types::gParamType_PrimVariant) {
        zeno::PrimVar var = zeno::reflect::any_cast<zeno::PrimVar>(param_defl);
        if (!std::holds_alternative<std::string>(var)) {
            return refSources;
        }
        std::string param_text = std::get<std::string>(var);
        if (param_text.find("ref(") == std::string::npos) {
            return refSources;
        }
        refSegments.push_back(param_text);
    }
    else if (deflType == zeno::types::gParamType_VecEdit) {
        const zeno::vecvar& vec = zeno::reflect::any_cast<zeno::vecvar>(param_defl);
        for (const zeno::PrimVar& elem : vec) {
            if (!std::holds_alternative<std::string>(elem)) {
                continue;
            }
            std::string param_text = std::get<std::string>(elem);
            if (param_text.find("ref(") != std::string::npos) {
                refSegments.push_back(param_text);
            }
        }
    }

    if (refSegments.empty())
        return refSources;

    auto namePath = get_path();

    //需要用zfxparser直接parse出所有引用信息
    GlobalError err;
    zeno::GraphException::catched([&] {
        auto& funcMgr = zeno::getSession().funcManager;
        ZfxContext ctx;
        ctx.spNode = shared_from_this();
        for (auto param_text : refSegments)
        {
            Formula fmla(param_text, namePath);
            int ret = fmla.parse();
            if (ret == 0)
            {
                ctx.code = param_text;
                std::shared_ptr<ZfxASTNode> astRoot = fmla.getASTResult();
                std::set<std::pair<std::string, std::string>> paths =
                    funcMgr->getReferSources(astRoot, &ctx);
                if (!paths.empty()) {
                    refSources.insert(paths.begin(), paths.end());
                }
            }
        }
    }, err);
    return refSources;
}

std::shared_ptr<DictObject> INode::processDict(ObjectParam* in_param) {
    std::shared_ptr<DictObject> spDict;
    //连接的元素是list还是list of list的规则，参照Graph::addLink下注释。
    bool bDirecyLink = false;
    const auto& inLinks = in_param->links;
    if (inLinks.size() == 1)
    {
        std::shared_ptr<ObjectLink> spLink = inLinks.front();
        auto out_param = spLink->fromparam;
        std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();

        if (out_param->type == in_param->type && spLink->tokey.empty()) //根据Graph::addLink规则，类型相同且无key视为直连
        {
            bDirecyLink = true;
            if (outNode->is_dirty()) {
                GraphException::translated([&] {
                    outNode->doApply();
                    }, outNode.get());

                auto outResult = outNode->get_output_obj(out_param->name);
                assert(outResult);
                assert(out_param->type == gParamType_Dict);

                if (in_param->socketType == Socket_Owning) {
                    spDict = std::dynamic_pointer_cast<DictObject>(outResult->move_clone());
                }
                else if (in_param->socketType == Socket_ReadOnly) {
                    spDict = std::dynamic_pointer_cast<DictObject>(outResult);
                }
                else if (in_param->socketType == Socket_Clone) {
                    //里面的元素也要clone
                    spDict = std::make_shared<DictObject>();
                    std::shared_ptr<DictObject> outDict = std::dynamic_pointer_cast<DictObject>(outResult);
                    for (auto& [key, spObject] : outDict->get()) {
                        //后续要考虑key的问题
                        spDict->lut.insert(std::make_pair(key, spObject->clone()));
                    }
                }
                spDict->update_key(m_uuid);
                return spDict;
            }
        }
    }
    if (!bDirecyLink)
    {
        spDict = std::make_shared<DictObject>();
        for (const auto& spLink : in_param->links)
        {
            const std::string& keyName = spLink->tokey;
            auto out_param = spLink->fromparam;
            std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();

            if (outNode->is_dirty()) {  //list中的元素是dirty的，重新计算并加入list
                GraphException::translated([&] {
                    outNode->doApply();
                    }, outNode.get());
            }

            auto outResult = outNode->get_output_obj(out_param->name);
            assert(outResult);
            if (in_param->socketType == Socket_Owning) {
                spDict->lut[keyName] = outResult->move_clone();
            }
            else if (in_param->socketType == Socket_ReadOnly) {
                spDict->lut[keyName] = outResult;
            }
            else if (in_param->socketType == Socket_Clone) {
                //后续要考虑key的问题
                spDict->lut[keyName] = outResult->clone();
            }
        }
        spDict->update_key(m_uuid);
        //已经是新构造的Dict了，不用复制了
    }
    return spDict;
}

std::shared_ptr<ListObject> INode::processList(ObjectParam* in_param) {
    std::shared_ptr<ListObject> spList;
    bool bDirectLink = false;
    if (in_param->links.size() == 1)
    {
        std::shared_ptr<ObjectLink> spLink = in_param->links.front();
        auto out_param = spLink->fromparam;
        std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();

        if (out_param->type == in_param->type && spLink->tokey.empty()) {   //根据Graph::addLink规则，类型相同且无key视为直连
            bDirectLink = true;

            if (outNode->is_dirty()) {
                GraphException::translated([&] {
                    outNode->doApply();
                    }, outNode.get());

                auto outResult = outNode->get_output_obj(out_param->name);
                assert(outResult);
                assert(out_param->type == gParamType_List);

                if (in_param->socketType == Socket_Owning) {
                    spList = std::dynamic_pointer_cast<ListObject>(outResult->move_clone());
                }
                else if (in_param->socketType == Socket_ReadOnly) {
                    spList = std::dynamic_pointer_cast<ListObject>(outResult);
                }
                else if (in_param->socketType == Socket_Clone) {
                    //里面的元素也要clone
                    spList = std::make_shared<ListObject>();
                    std::shared_ptr<ListObject> outList = std::dynamic_pointer_cast<ListObject>(outResult);
                    for (int i = 0; i < outList->size(); i++) {
                        //后续要考虑key的问题
                        spList->push_back(outList->get(i)->clone());
                    }
                }
                spList->update_key(m_uuid);
            }
        }
    }
    if (!bDirectLink)
    {
        spList = std::make_shared<ListObject>();
        int indx = 0;
        for (const auto& spLink : in_param->links)
        {
            //list的情况下，keyName是不是没意义，顺序怎么维持？
            auto out_param = spLink->fromparam;
            std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();
            if (outNode->is_dirty()) {  //list中的元素是dirty的，重新计算并加入list
                GraphException::translated([&] {
                    outNode->doApply();
                    }, outNode.get());
            }
            auto outResult = outNode->get_output_obj(out_param->name);
            assert(outResult);

            if (in_param->socketType == Socket_Owning) {
                spList->push_back(outResult->move_clone());
            }
            else if (in_param->socketType == Socket_ReadOnly) {
                spList->push_back(outResult);
            }
            else if (in_param->socketType == Socket_Clone) {
                spList->push_back(outResult->clone());
            }
            //spList->dirtyIndice.insert(indx);
        }
        spList->update_key(m_uuid);
    }
    return spList;
}

zeno::reflect::Any INode::processPrimitive(PrimitiveParam* in_param)
{
    if (!in_param) {
        return nullptr;
    }

    //观察参数是否存在引用参数
    auto refSources = resolveReferSource(in_param->name);
    //如果有，需要先执行，因为我们需要获得参数的实际计算值
    for (auto refSource : refSources)
    {
        const std::string& source_node_uuidpath = refSource.first;
        auto spNode = getSession().mainGraph->getNodeByUuidPath(source_node_uuidpath);
        assert(spNode);
        GraphException::translated([&] {
            spNode->doApply();
        }, spNode.get());
        //后续执行ref函数的时候，上述节点的计算结果已经出来了
    }

    int frame = getGlobalState()->getFrameId();
    //zany result;

    const ParamType type = in_param->type;
    const auto& defl = in_param->defl;
    zeno::reflect::Any result = defl;
    ParamType editType = defl.type().hash_code();

    switch (type) {
    case gParamType_Int:
    case gParamType_Float:
    {
        if (editType == gParamType_PrimVariant) {
            zeno::PrimVar var = any_cast<zeno::PrimVar>(defl);
            result = std::visit([=](auto&& arg)->Any {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
                    return arg;
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    return resolve(arg, type);
                }
                else if constexpr (std::is_same_v<T, CurveData>) {
                    int frame = getGlobalState()->getFrameId();
                    return arg.eval(frame);
                }
                else {
                    throw UnimplError();
                }
            }, var);
        }
        else if (editType == gParamType_Int) {
            //目前所有defl都是以PrimVariant的方式储存，暂时不会以本值类型储存
            assert(false);
        }
        else if (editType == gParamType_Float) {
            assert(false);
        }
        else {
            assert(false);
        }
        break;
    }
    case zeno::types::gParamType_Bool:
    {
        //Bool值暂不支持控件编写表达式，因此直接取值
        assert(editType == gParamType_Bool);
        result = std::move(defl);
        break;
    }
    case zeno::types::gParamType_String:
    {
        //TODO: format string as formula
        break;
    }
    case gParamType_Vec2f:
    case gParamType_Vec2i:
    case gParamType_Vec3f:
    case gParamType_Vec3i:
    case gParamType_Vec4f:
    case gParamType_Vec4i:
    {
        assert(gParamType_VecEdit == editType);
        zeno::vecvar editvec = any_cast<zeno::vecvar>(defl);
        std::vector<float> vec;
        for (int i = 0; i < editvec.size(); i++)
        {
            float res = std::visit([=](auto&& arg)->float {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
                    return arg;
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    return resolve(arg, type);
                }
                else if constexpr (std::is_same_v<T, CurveData>) {
                    int frame = getGlobalState()->getFrameId();
                    return arg.eval(frame);
                }
                else {
                    throw UnimplError();
                }
            }, editvec[i]);
            vec.push_back(res);
        }
        if (type == gParamType_Vec2f)       result = zeno::vec2f(vec[0], vec[1]);
        else if (type == gParamType_Vec2i)  result = zeno::vec2i(vec[0], vec[1]);
        else if (type == gParamType_Vec3f)  result = zeno::vec3f(vec[0], vec[1], vec[2]);
        else if (type == gParamType_Vec3i)  result = zeno::vec3i(vec[0], vec[1], vec[2]);
        else if (type == gParamType_Vec4f)  result = zeno::vec4f(vec[0], vec[1], vec[2], vec[3]);
        else if (type == gParamType_Vec4i)  result = zeno::vec4i(vec[0], vec[1], vec[2], vec[3]);
        break;
    }
    case zeno::types::gParamType_Heatmap:
    {
        //TODO: heatmap的结构体定义.
        //if (std::holds_alternative<std::string>(defl))
        //    result = zeno::parseHeatmapObj(std::get<std::string>(defl));
        break;
    }
    //这里指的是基础类型的List/Dict.
    case gParamType_List:
    {
        //TODO: List现在还没有ui支持，而且List是泛型容器，对于非Literal值不好设定默认值。
        break;
    }
    case gParamType_Dict:
    {
        break;
    }
    }
    return result;
}

bool INode::receiveOutputObj(ObjectParam* in_param, zany outputObj, ParamType outobj_type) {

    if (in_param->socketType == Socket_Clone) {
        in_param->spObject = outputObj->clone();
    }
    else if (in_param->socketType == Socket_Owning) {
        in_param->spObject = outputObj->move_clone();
    }
    else if (in_param->socketType == Socket_ReadOnly) {
        in_param->spObject = outputObj;
        //TODO: readonly property on object.
    }
    return true;
}

ZENO_API bool INode::requireInput(std::string const& ds) {
    // 目前假设输入对象和输入数值，不能重名（不难实现，老节点直接改）。
    auto iter = m_inputObjs.find(ds);
    if (iter != m_inputObjs.end()) {
        ObjectParam* in_param = &(iter->second);
        if (in_param->links.empty()) {
            //节点如果定义了对象，但没有边连上去，是否要看节点apply如何处理？
            //FIX: 没有边的情况要清空掉对象，否则apply以为这个参数连上了对象
            in_param->spObject.reset();
        }
        else {
            switch (in_param->type)
            {
                case gParamType_Dict:
                {
                    std::shared_ptr<DictObject> outDict = processDict(in_param);
                    receiveOutputObj(in_param, outDict, gParamType_Dict);
                    break;
                }
                case gParamType_List:
                {
                    std::shared_ptr<ListObject> outList = processList(in_param);
                    receiveOutputObj(in_param, outList, gParamType_List);
                    break;
                }
                case gParamType_Curve:
                {
                    //Curve要视作Object，因为整合到variant太麻烦，只要对于最原始的MakeCurve节点，以字符串（储存json）作为特殊类型即可。
                }
                default:
                {
                    if (in_param->links.size() == 1)
                    {
                        std::shared_ptr<ObjectLink> spLink = *(in_param->links.begin());
                        ObjectParam* out_param = spLink->fromparam;
                        std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();

                        GraphException::translated([&] {
                            outNode->doApply();
                        }, outNode.get());

                        if (out_param->spObject)
                        {
                            receiveOutputObj(in_param, out_param->spObject, out_param->type);
                        }
                    }
                }
            }
        }
    }
    else {
        auto iter2 = m_inputPrims.find(ds);
        if (iter2 != m_inputPrims.end()) {
            PrimitiveParam* in_param = &iter2->second;
            if (in_param->links.empty()) {
                in_param->result = processPrimitive(in_param);
                //旧版本的requireInput指的是是否有连线，如果想兼容旧版本，这里可以返回false，但使用量不多，所以就修改它的定义。
            }
            else {
                if (in_param->links.size() == 1) {
                    std::shared_ptr<PrimitiveLink> spLink = *in_param->links.begin();
                    std::shared_ptr<INode> outNode = spLink->fromparam->m_wpNode.lock();

                    GraphException::translated([&] {
                        outNode->doApply();
                    }, outNode.get());
                    //数值基本类型，直接复制。
                    in_param->result = spLink->fromparam->result;
                }
            }
        } else {
            return false;
        }
    }
    return true;
}

ZENO_API void INode::doOnlyApply() {
    apply();
}

ZENO_API void INode::doApply() {

    if (!m_dirty) {
        registerObjToManager();//如果只是打view，也是需要加到manager的。
        return;
    }

    /*
    zeno::scope_exit spe([&] {//apply时根据情况将IParam标记为modified，退出时将所有IParam标记为未modified
        for (auto const& [name, param] : m_outputs)
            param.m_idModify = false;
        });
    */

    preApply();

    if (zeno::getSession().is_interrupted()) {
        throw makeError<InterruputError>(m_uuidPath);
    }

    log_debug("==> enter {}", m_name);
    {
#ifdef ZENO_BENCHMARKING
        Timer _(m_name);
#endif
        reportStatus(true, Node_Running);
        if (!m_pTypebase) {
            apply();
        }
        else {
            reflecNode_apply();
        }
    }
    log_debug("==> leave {}", m_name);

    registerObjToManager();
    reportStatus(false, Node_RunSucceed);
}

ZENO_API ObjectParams INode::get_input_object_params() const
{
    ObjectParams params;
    for (auto& [name, spObjParam] : m_inputObjs)
    {
        ParamObject obj;
        for (auto linkInfo : spObjParam.links) {
            obj.links.push_back(getEdgeInfo(linkInfo));
        }
        obj.name = name;
        obj.type = spObjParam.type;
        obj.bInput = true;
        obj.socketType = spObjParam.socketType;
        obj.wildCardGroup = spObjParam.wildCardGroup;
        //obj.prop = ?
        params.push_back(obj);
    }
    return params;
}

ZENO_API ObjectParams INode::get_output_object_params() const
{
    ObjectParams params;
    for (auto& [name, spObjParam] : m_outputObjs)
    {
        ParamObject obj;
        for (auto linkInfo : spObjParam.links) {
            obj.links.push_back(getEdgeInfo(linkInfo));
        }
        obj.name = name;
        obj.type = spObjParam.type;
        obj.bInput = false;
        obj.socketType = spObjParam.socketType;
        obj.wildCardGroup = spObjParam.wildCardGroup;
        //obj.prop = ?
        params.push_back(obj);
    }
    return params;
}

ZENO_API PrimitiveParams INode::get_input_primitive_params() const {
    //TODO: deprecated node.
    PrimitiveParams params;
    for (auto& [name, spParamObj] : m_inputPrims) {
        ParamPrimitive param;
        param.bInput = true;
        param.name = name;
        param.type = spParamObj.type;
        param.control = spParamObj.control;
        param.ctrlProps = spParamObj.ctrlProps;
        param.defl = spParamObj.defl;
        param.bVisible = spParamObj.bVisible;
        for (auto spLink : spParamObj.links) {
            param.links.push_back(getEdgeInfo(spLink));
        }
        param.socketType = spParamObj.socketType;
        param.wildCardGroup = spParamObj.wildCardGroup;
        params.push_back(param);
    }
    return params;
}

ZENO_API PrimitiveParams INode::get_output_primitive_params() const {
    PrimitiveParams params;
    for (auto& [name, spParamObj] : m_outputPrims) {
        ParamPrimitive param;
        param.bInput = false;
        param.name = name;
        param.type = spParamObj.type;
        param.control = NullControl;
        param.defl = spParamObj.defl;
        for (auto spLink : spParamObj.links) {
            param.links.push_back(getEdgeInfo(spLink));
        }
        param.socketType = spParamObj.socketType;
        param.wildCardGroup = spParamObj.wildCardGroup;
        params.push_back(param);
    }
    return params;
}

ZENO_API ParamPrimitive INode::get_input_prim_param(std::string const& name, bool* pExist) const {
    ParamPrimitive param;
    auto iter = m_inputPrims.find(name);
    if (iter != m_inputPrims.end()) {
        auto& paramPrim = iter->second;
        param = paramPrim.exportParam();
        if (pExist)
            *pExist = true;
    }
    else {
        if (pExist)
            *pExist = false;
    }
    return param;
}

ZENO_API ParamObject INode::get_input_obj_param(std::string const& name, bool* pExist) const {
    ParamObject param;
    auto iter = m_inputObjs.find(name);
    if (iter != m_inputObjs.end()) {
        auto& paramObj = iter->second;
        param = paramObj.exportParam();
        if (pExist)
            *pExist = true;
    }
    else {
        if (pExist)
            *pExist = false;
    }
    return param;
}

ZENO_API ParamPrimitive INode::get_output_prim_param(std::string const& name, bool* pExist) const {
    ParamPrimitive param;
    auto iter = m_outputPrims.find(name);
    if (iter != m_outputPrims.end()) {
        auto& paramPrim = iter->second;
        param = paramPrim.exportParam();
        if (pExist)
            *pExist = true;
    }
    else {
        if (pExist)
            *pExist = false;
    }
    return param;
}

ZENO_API ParamObject INode::get_output_obj_param(std::string const& name, bool* pExist) const {
    ParamObject param;
    auto iter = m_outputObjs.find(name);
    if (iter != m_outputObjs.end()) {
        auto& paramObj = iter->second;
        param = paramObj.exportParam();
        if (pExist)
            *pExist = true;
    }
    else {
        if (pExist)
            *pExist = false;
    }
    return param;
}

bool INode::add_input_prim_param(ParamPrimitive param) {
    if (m_inputPrims.find(param.name) != m_inputPrims.end()) {
        return false;
    }
    PrimitiveParam sparam;
    sparam.bInput = true;
    sparam.control = param.control;
    sparam.defl = param.defl;
    convertToEditVar(sparam.defl, param.type);
    sparam.m_wpNode = shared_from_this();
    sparam.name = param.name;
    sparam.socketType = param.socketType;
    sparam.type = param.type;
    sparam.ctrlProps = param.ctrlProps;
    sparam.bVisible = param.bVisible;
    sparam.wildCardGroup = param.wildCardGroup;
    m_inputPrims.insert(std::make_pair(param.name, std::move(sparam)));
    return true;
}

bool INode::add_input_obj_param(ParamObject param) {
    if (m_inputObjs.find(param.name) != m_inputObjs.end()) {
        return false;
    }
    ObjectParam sparam;
    sparam.bInput = true;
    sparam.name = param.name;
    sparam.type = param.type;
    sparam.socketType = param.socketType;
    sparam.m_wpNode = shared_from_this();
    sparam.wildCardGroup = param.wildCardGroup;
    m_inputObjs.insert(std::make_pair(param.name, std::move(sparam)));
    return true;
}

bool INode::add_output_prim_param(ParamPrimitive param) {
    if (m_outputPrims.find(param.name) != m_outputPrims.end()) {
        return false;
    }
    PrimitiveParam sparam;
    sparam.bInput = false;
    sparam.control = param.control;
    sparam.defl = param.defl;
    sparam.m_wpNode = shared_from_this();
    sparam.name = param.name;
    sparam.socketType = param.socketType;
    sparam.type = param.type;
    sparam.ctrlProps = param.ctrlProps;
    sparam.wildCardGroup = param.wildCardGroup;
    sparam.bVisible = param.bVisible;
    m_outputPrims.insert(std::make_pair(param.name, std::move(sparam)));
    return true;
}

bool INode::add_output_obj_param(ParamObject param) {
    if (m_outputObjs.find(param.name) != m_outputObjs.end()) {
        return false;
    }
    ObjectParam sparam;
    sparam.bInput = false;
    sparam.name = param.name;
    sparam.type = param.type;
    sparam.socketType = param.socketType;
    sparam.m_wpNode = shared_from_this();
    sparam.wildCardGroup = param.wildCardGroup;
    m_outputObjs.insert(std::make_pair(param.name, std::move(sparam)));
    return true;
}

ZENO_API void INode::set_result(bool bInput, const std::string& name, zany spObj) {
    if (bInput) {
        auto& param = safe_at(m_inputObjs, name, "");
        param.spObject = spObj;
    }
    else {
        auto& param = safe_at(m_outputObjs, name, "");
        param.spObject = spObj;
    }
}

void INode::init_object_link(bool bInput, const std::string& paramname, std::shared_ptr<ObjectLink> spLink, const std::string& targetParam) {
    auto iter = bInput ? m_inputObjs.find(paramname) : m_outputObjs.find(paramname);
    if (bInput)
        spLink->toparam = &iter->second;
    else
        spLink->fromparam = &iter->second;
    spLink->targetParam = targetParam;
    iter->second.links.emplace_back(spLink);
}

void INode::init_primitive_link(bool bInput, const std::string& paramname, std::shared_ptr<PrimitiveLink> spLink, const std::string& targetParam) {
    auto iter = bInput ? m_inputPrims.find(paramname) : m_outputPrims.find(paramname);
    if (bInput)
        spLink->toparam = &iter->second;
    else
        spLink->fromparam = &iter->second;
    spLink->targetParam = targetParam;
    iter->second.links.emplace_back(spLink);
}

bool INode::isPrimitiveType(bool bInput, const std::string& param_name, bool& bExist) {
    if (bInput) {
        if (m_inputObjs.find(param_name) != m_inputObjs.end()) {
            bExist = true;
            return false;
        }
        else if (m_inputPrims.find(param_name) != m_inputPrims.end()) {
            bExist = true;
            return true;
        }
        bExist = false;
        return false;
    }
    else {
        if (m_outputObjs.find(param_name) != m_outputObjs.end()) {
            bExist = true;
            return false;
        }
        else if (m_outputPrims.find(param_name) != m_outputPrims.end()) {
            bExist = true;
            return true;
        }
        bExist = false;
        return false;
    }
}

std::vector<EdgeInfo> INode::getLinks() const {
    std::vector<EdgeInfo> remLinks;
    for (const auto& [_, spParam] : m_inputObjs) {
        for (std::shared_ptr<ObjectLink> spLink : spParam.links) {
            remLinks.push_back(getEdgeInfo(spLink));
        }
    }
    for (const auto& [_, spParam] : m_inputPrims) {
        for (std::shared_ptr<PrimitiveLink> spLink : spParam.links) {
            remLinks.push_back(getEdgeInfo(spLink));
        }
    }
    for (const auto& [_, spParam] : m_outputObjs) {
        for (std::shared_ptr<ObjectLink> spLink : spParam.links) {
            remLinks.push_back(getEdgeInfo(spLink));
        }
    }
    for (const auto& [_, spParam] : m_outputPrims) {
        for (std::shared_ptr<PrimitiveLink> spLink : spParam.links) {
            remLinks.push_back(getEdgeInfo(spLink));
        }
    }
    return remLinks;
}

std::vector<EdgeInfo> INode::getLinksByParam(bool bInput, const std::string& param_name) const {
    std::vector<EdgeInfo> links;

    auto& objects = bInput ? m_inputObjs : m_outputObjs;
    auto& primtives = bInput ? m_inputPrims : m_outputPrims;

    auto iter = objects.find(param_name);
    if (iter != objects.end()) {
        for (auto spLink : iter->second.links) {
            links.push_back(getEdgeInfo(spLink));
        }
    }
    else {
        auto iter2 = primtives.find(param_name);
        if (iter2 != primtives.end()) {
            for (auto spLink : iter2->second.links) {
                links.push_back(getEdgeInfo(spLink));
            }
        }
    }
    return links;
}

bool INode::updateLinkKey(bool bInput, const zeno::EdgeInfo& edge, const std::string& oldkey, const std::string& newkey)
{
    auto& objects = bInput ? m_inputObjs : m_outputObjs;
    auto iter = objects.find(edge.inParam);
    if (iter != objects.end()) {
        for (auto spLink : iter->second.links) {
            if (auto fromParam = spLink->fromparam) {
                if (auto outnode = fromParam->m_wpNode.lock()) {
                    if (outnode->get_name() == edge.outNode && spLink->tokey == oldkey) {   //需outnode和tokey均相同
                        spLink->tokey = newkey;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool INode::moveUpLinkKey(bool bInput, const std::string& param_name, const std::string& key)
{
    auto& objects = bInput ? m_inputObjs : m_outputObjs;
    auto iter = objects.find(param_name);
    if (iter != objects.end()) {
        for (auto it = iter->second.links.begin(); it != iter->second.links.end(); it++) {
            if ((*it)->tokey == key) {
                auto it_ = std::prev(it);
                std::swap(*it, *it_);
                return true;
            }
        }
    }
    return false;
}

bool INode::removeLink(bool bInput, const EdgeInfo& edge) {
    if (bInput) {
        if (edge.bObjLink) {
            auto iter = m_inputObjs.find(edge.inParam);
            if (iter == m_inputObjs.end())
                return false;
            for (auto spLink : iter->second.links) {
                if (auto outNode = spLink->fromparam->m_wpNode.lock()) {
                    if (outNode->get_name() == edge.outNode && spLink->fromparam->name == edge.outParam && spLink->fromkey == edge.outKey) {
                        iter->second.links.remove(spLink);
                    return true;
                }
            }
        }
        }
        else {
            auto iter = m_inputPrims.find(edge.inParam);
            if (iter == m_inputPrims.end())
                return false;
            for (auto spLink : iter->second.links) {
                if (auto outNode = spLink->fromparam->m_wpNode.lock()) {
                    if (outNode->get_name() == edge.outNode && spLink->fromparam->name == edge.outParam) {
                        iter->second.links.remove(spLink);
                    return true;
                }
            }
        }
    }
    }
    else {
        if (edge.bObjLink) {
            auto iter = m_outputObjs.find(edge.outParam);
            if (iter == m_outputObjs.end())
                return false;
            for (auto spLink : iter->second.links) {
                if (auto inNode = spLink->toparam->m_wpNode.lock()) {
                    if (inNode->get_name() == edge.inNode && spLink->toparam->name == edge.inParam && spLink->tokey == edge.inKey) {
                        iter->second.links.remove(spLink);
                    return true;
                }
            }
        }
        }
        else {
            auto iter = m_outputPrims.find(edge.outParam);
            if (iter == m_outputPrims.end())
                return false;
            for (auto spLink : iter->second.links) {
                if (auto inNode = spLink->toparam->m_wpNode.lock()) {
                    if (inNode->get_name() == edge.inNode && spLink->toparam->name == edge.inParam) {
                        iter->second.links.remove(spLink);
                    return true;
                }
            }
        }
    }
    }
    return false;
}

ZENO_API std::string INode::get_viewobject_output_param() const {
    //现在暂时还没有什么标识符用于指定哪个输出口是对应输出view obj的
    //一般都是默认第一个输出obj，暂时这么规定，后续可能用标识符。
    if (m_outputObjs.empty())
        return "";
    return m_outputObjs.begin()->second.name;
}

ZENO_API NodeData INode::exportInfo() const
{
    NodeData node;
    node.cls = m_nodecls;
    node.name = m_name;
    node.bView = m_bView;
    node.uipos = m_pos;
    //TODO: node type
    if (node.subgraph.has_value())
        node.type = Node_SubgraphNode;
    else
        node.type = Node_Normal;

    node.customUi = get_customui();
    node.customUi.inputObjs.clear();
    for (auto& [name, paramObj] : m_inputObjs)
    {
        node.customUi.inputObjs.push_back(paramObj.exportParam());
    }
    if (m_nodecls == "SubOutput") {     //SubOutput节点tabs-groups-params为空，需单独导出primitiveInputs
        if (!node.customUi.inputPrims.tabs.empty() && !node.customUi.inputPrims.tabs[0].groups.empty()) {
            for (auto& [name, paramPrimitive] : m_inputPrims) {
                node.customUi.inputPrims.tabs[0].groups[0].params.push_back(paramPrimitive.exportParam());
            }
        }
    }
    else {
    for (auto &tab : node.customUi.inputPrims.tabs)
    {
        for (auto &group : tab.groups)
        {
            for (auto& param : group.params)
            {
                auto iter = m_inputPrims.find(param.name);
                if (iter != m_inputPrims.end())
                {
                    param = iter->second.exportParam();
                }
            }
        }
    }
    }

    node.customUi.outputPrims.clear();
    for (auto& [name, paramObj] : m_outputPrims)
    {
        node.customUi.outputPrims.push_back(paramObj.exportParam());
    }
    node.customUi.outputObjs.clear();
    for (auto& [name, paramObj] : m_outputObjs)
    {
        node.customUi.outputObjs.push_back(paramObj.exportParam());
    }
    return node;
}

ZENO_API bool INode::update_param(const std::string& param, zeno::reflect::Any new_value) {
    CORE_API_BATCH
    auto& spParam = safe_at(m_inputPrims, param, "miss input param `" + param + "` on node `" + m_name + "`");
    bool isvalid = convertToEditVar(new_value, spParam.type);
    if (!isvalid) {
        zeno::log_error("cannot convert to edit variable");
        return false;
    }
    if (!isAnyEqual(spParam.defl, new_value))
    {
        auto old_value = spParam.defl;
        spParam.defl = new_value;

        std::shared_ptr<Graph> spGraph = graph.lock();
        assert(spGraph);

        spGraph->onNodeParamUpdated(&spParam, old_value, new_value);
        CALLBACK_NOTIFY(update_param, param, old_value, new_value)
        mark_dirty(true);

        auto& refMgr = getSession().referManager;
        auto refSources = resolveReferSource(spParam.name);
        if (!refSources.empty()) {
            refMgr->registerRelations(m_uuidPath, spParam.name, refSources);
        }

        //当前节点的这个参数可能被引用，需要找到所有引用者并标脏它们
        refMgr->updateDirty(m_uuidPath, spParam.name);
        return true;
    }
    return false;
}

ZENO_API bool zeno::INode::update_param_socket_type(const std::string& param, SocketType type)
{
    CORE_API_BATCH
    auto& spParam = safe_at(m_inputObjs, param, "miss input param `" + param + "` on node `" + m_name + "`");
    if (type != spParam.socketType)
    {
        spParam.socketType = type;
        if (type == Socket_Owning)
        {
            auto spGraph = graph.lock();
            spGraph->removeLinks(m_name, true, param);
        }
        mark_dirty(true);
        CALLBACK_NOTIFY(update_param_socket_type, param, type)
        return true;
    }
    return false;
}

ZENO_API bool zeno::INode::update_param_type(const std::string& param, bool bPrim, bool bInput, ParamType type)
{
    CORE_API_BATCH
        if (bPrim)
        {
            auto& prims = bInput ? m_inputPrims : m_outputPrims;
            auto& prim = prims.find(param);
            if (prim != prims.end())
            {
                auto& spParam = prim->second;
                if (type != spParam.type)
    {
                    spParam.type = type;
                    CALLBACK_NOTIFY(update_param_type, param, type)
                        return true;
                }
            }
        }
        else 
        {
            auto& objects = bInput ? m_inputObjs : m_outputObjs;
            auto& object = objects.find(param);
            if (object != objects.end())
            {
                auto& spParam = object->second;
                if (type != spParam.type)
                {
                    spParam.type = type;
                    CALLBACK_NOTIFY(update_param_type, param, type)
                    return true;
    }
            }
        }
    return false;
}

ZENO_API bool zeno::INode::update_param_control(const std::string& param, ParamControl control)
{
    CORE_API_BATCH
    auto& spParam = safe_at(m_inputPrims, param, "miss input param `" + param + "` on node `" + m_name + "`");
    if (control != spParam.control)
    {
        spParam.control = control;
        CALLBACK_NOTIFY(update_param_control, param, control)
        return true;
    }
    return false;
}

ZENO_API bool zeno::INode::update_param_control_prop(const std::string& param, zeno::reflect::Any props)
{
    CORE_API_BATCH
    auto& spParam = safe_at(m_inputPrims, param, "miss input param `" + param + "` on node `" + m_name + "`");
    spParam.ctrlProps = props;
        CALLBACK_NOTIFY(update_param_control_prop, param, props)
        return true;
}

ZENO_API bool zeno::INode::update_param_visible(const std::string& param, bool bVisible, bool bInput)
{
    CORE_API_BATCH
    if (bInput) {
    auto& spParam = safe_at(m_inputPrims, param, "miss input param `" + param + "` on node `" + m_name + "`");

        if (spParam.bVisible != bVisible)
    {
            spParam.bVisible = bVisible;
            CALLBACK_NOTIFY(update_param_visible, param, bVisible, bInput)
            return true;
    }
    } else {
        auto& spParam = safe_at(m_outputPrims, param, "miss output param `" + param + "` on node `" + m_name + "`");

        if (spParam.bVisible != bVisible)
        {
            spParam.bVisible = bVisible;
            CALLBACK_NOTIFY(update_param_visible, param, bVisible, bInput)
                return true;
        }
    }
    return false;
}

ZENO_API void INode::update_param_color(const std::string& name, std::string& clr)
{
    CORE_API_BATCH
    CALLBACK_NOTIFY(update_param_color, name, clr)
}

ZENO_API void INode::update_layout(params_change_info& changes)
{
    CALLBACK_NOTIFY(update_layout, changes);
}

ZENO_API params_change_info INode::update_editparams(const ParamsUpdateInfo& params)
{
    //TODO: 这里只有primitive参数类型的情况，还需要整合obj参数的情况。
    std::set<std::string> inputs_old, outputs_old, obj_inputs_old, obj_outputs_old;
    for (const auto& [param_name, _] : m_inputPrims) {
        inputs_old.insert(param_name);
    }
    for (const auto& [param_name, _] : m_outputPrims) {
        outputs_old.insert(param_name);
    }
    for (const auto& [param_name, _] : m_inputObjs) {
        obj_inputs_old.insert(param_name);
    }
    for (const auto& [param_name, _] : m_outputObjs) {
        obj_outputs_old.insert(param_name);
    }

    params_change_info changes;
    std::set<std::tuple<std::string, zeno::ParamType, bool>> paramTypeChanges;

    for (auto _pair : params) {
        if (const auto& pParam = std::get_if<ParamObject>(&_pair.param))
        {
            const ParamObject& param = *pParam;
            const std::string oldname = _pair.oldName;
            const std::string newname = param.name;

            auto& in_outputs = param.bInput ? m_inputObjs : m_outputObjs;
            auto& new_params = param.bInput ? changes.new_inputs : changes.new_outputs;
            auto& remove_params = param.bInput ? changes.remove_inputs : changes.remove_outputs;
            auto& rename_params = param.bInput ? changes.rename_inputs : changes.rename_outputs;

            if (oldname.empty()) {
                //new added name.
                if (in_outputs.find(newname) != in_outputs.end()) {
                    // the new name happen to have the same name with the old name, but they are not the same param.
                    in_outputs.erase(newname);
                    if (param.bInput)
                        obj_inputs_old.erase(newname);
                    else
                        obj_outputs_old.erase(newname);

                    remove_params.insert(newname);
                }

                ObjectParam sparam;
                sparam.name = newname;
                sparam.type = param.type;
                sparam.socketType = param.socketType;
                sparam.m_wpNode = shared_from_this();
                in_outputs[newname] = std::move(sparam);

                new_params.insert(newname);
            }
            else if (in_outputs.find(oldname) != in_outputs.end()) {
                if (oldname != newname) {
                    //exist name changed.
                    in_outputs[newname] = std::move(in_outputs[oldname]);
                    in_outputs.erase(oldname);

                    rename_params.insert({ oldname, newname });
                }
                else {
                    //name stays.
                }

                if (param.bInput)
                    obj_inputs_old.erase(oldname);
                else
                    obj_outputs_old.erase(oldname);

                auto& spParam = in_outputs[newname];
                spParam.type = param.type;
                spParam.name = newname;
                if (param.bInput)
                {
                    update_param_socket_type(spParam.name, param.socketType);
                }
            }
            else {
                throw makeError<KeyError>(oldname, "the name does not exist on the node");
            }
        }
        else if (const auto& pParam = std::get_if<ParamPrimitive>(&_pair.param))
        {
            const ParamPrimitive& param = *pParam;
            const std::string oldname = _pair.oldName;
            const std::string newname = param.name;

            auto& in_outputs = param.bInput ? m_inputPrims : m_outputPrims;
            auto& new_params = param.bInput ? changes.new_inputs : changes.new_outputs;
            auto& remove_params = param.bInput ? changes.remove_inputs : changes.remove_outputs;
            auto& rename_params = param.bInput ? changes.rename_inputs : changes.rename_outputs;

            if (oldname.empty()) {
                //new added name.
                if (in_outputs.find(newname) != in_outputs.end()) {
                    // the new name happen to have the same name with the old name, but they are not the same param.
                    in_outputs.erase(newname);
                    if (param.bInput)
                        inputs_old.erase(newname);
                    else
                        outputs_old.erase(newname);

                    remove_params.insert(newname);
                }

                PrimitiveParam sparam;
                sparam.defl = param.defl;
                convertToEditVar(sparam.defl, param.type);
                sparam.name = newname;
                sparam.type = param.type;
                sparam.control = param.control;
                sparam.ctrlProps = param.ctrlProps;
                sparam.socketType = param.socketType;
                sparam.m_wpNode = shared_from_this();
                sparam.bVisible = param.bVisible;
                in_outputs[newname] = std::move(sparam);

                new_params.insert(newname);
            }
            else if (in_outputs.find(oldname) != in_outputs.end()) {
                if (oldname != newname) {
                    //exist name changed.
                    in_outputs[newname] = std::move(in_outputs[oldname]);
                    in_outputs.erase(oldname);

                    rename_params.insert({ oldname, newname });
                }
                else {
                    //name stays.
                }

                if (param.bInput)
                    inputs_old.erase(oldname);
                else
                    outputs_old.erase(oldname);

                auto& spParam = in_outputs[newname];
                spParam.defl = param.defl;
                convertToEditVar(spParam.defl, spParam.type);
                spParam.name = newname;
                spParam.socketType = param.socketType;
                if (param.bInput)
                {
                    if (param.type != spParam.type)
                    {
                        paramTypeChanges.insert({newname, param.type, param.bInput});
                        //update_param_type(spParam->name, param.type);
                        //if (auto spNode = subgraph->getNode(oldname))
                        //    spNode->update_param_type("port", param.type);
                    }
                    update_param_control(spParam.name, param.control);
                    update_param_control_prop(spParam.name, param.ctrlProps);
                }
            }
            else {
                throw makeError<KeyError>(oldname, "the name does not exist on the node");
            }
        }
    }

    std::shared_ptr<Graph> spGraph = graph.lock();

    //the left names are the names of params which will be removed.
    for (auto rem_name : inputs_old) {
        if (spGraph)
            spGraph->removeLinks(get_name(), true, rem_name);
        m_inputPrims.erase(rem_name);
        changes.remove_inputs.insert(rem_name);
    }

    for (auto rem_name : outputs_old) {
        if (spGraph)
            spGraph->removeLinks(get_name(), false, rem_name);
        m_outputPrims.erase(rem_name);
        changes.remove_outputs.insert(rem_name);
    }

    for (auto rem_name : obj_inputs_old) {
        if (spGraph)
            spGraph->removeLinks(get_name(), true, rem_name);
        m_inputObjs.erase(rem_name);
        changes.remove_inputs.insert(rem_name);
    }

    for (auto rem_name : obj_outputs_old) {
        if (spGraph)
            spGraph->removeLinks(get_name(), false, rem_name);
        m_outputObjs.erase(rem_name);
        changes.remove_outputs.insert(rem_name);
    }
    changes.inputs.clear();
    changes.outputs.clear();
    for (const auto& [param, _] : params) {
        if (auto paramPrim = std::get_if<ParamPrimitive>(&param))
        {
            if (paramPrim->bInput)
                changes.inputs.push_back(paramPrim->name);
            else
                changes.outputs.push_back(paramPrim->name);
        }
        else if (auto paramPrim = std::get_if<ParamObject>(&param))
        {
            if (paramPrim->bInput)
                changes.inputs.push_back(paramPrim->name);
            else
                changes.outputs.push_back(paramPrim->name);
        }
    }

    for (const auto& [name, type, bInput] : paramTypeChanges) {
        update_param_type(name, true, bInput, type);
    }
    return changes;
}

ZENO_API void INode::trigger_update_params(const std::string& param, bool changed, params_change_info changes)
{
    if (changed)
        update_layout(changes);
}

ZENO_API void INode::init(const NodeData& dat)
{
    //IO init
    if (!dat.name.empty())
        m_name = dat.name;

    if (m_name == "selfinc") {
        int j;
        j = -0;
    }

    m_pos = dat.uipos;
    m_bView = dat.bView;
    if (m_bView) {
        std::shared_ptr<Graph> spGraph = graph.lock();
        assert(spGraph);
        spGraph->viewNodeUpdated(m_name, m_bView);
    }
    if (SubnetNode* pSubnetNode = dynamic_cast<SubnetNode*>(this))
    {
        pSubnetNode->setCustomUi(dat.customUi);
    }
    initParams(dat);
    m_dirty = true;
}

ZENO_API void INode::initParams(const NodeData& dat)
{
    for (const ParamObject& paramObj : dat.customUi.inputObjs)
    {
        auto iter = m_inputObjs.find(paramObj.name);
        if (iter == m_inputObjs.end()) {
            add_input_obj_param(paramObj);
            continue;
        }
        auto& sparam = iter->second;

        //如果不是子图，不能读写socketType，一律以节点定义处为准。（TODO: 如果涉及到转为owning，甚至有些obj连线要去掉）
        if (dat.type == Node_SubgraphNode || dat.type == Node_AssetInstance) {
            sparam.socketType = paramObj.socketType;
        }
    }
    for (auto tab : dat.customUi.inputPrims.tabs)
    {
        for (auto group : tab.groups)
        {
            for (auto param : group.params)
            {
                auto iter = m_inputPrims.find(param.name);
                if (iter == m_inputPrims.end()) {
                    add_input_prim_param(param);
                    continue;
                }
                auto& sparam = iter->second;
                sparam.defl = param.defl;
                convertToEditVar(sparam.defl, param.type);
                sparam.control = param.control;
                sparam.ctrlProps = param.ctrlProps;
                sparam.bVisible = param.bVisible;
                sparam.type = param.type;

                //resolve引用关系
                //auto& refMgr = getSession().referManager;
                //auto refSources = resolveReferSource(sparam.name);
                //if (!refSources.empty()) {
                //    refMgr->registerRelations(m_uuidPath, sparam.name, refSources);
                //}

                //graph记录$F相关节点
                if (std::shared_ptr<Graph> spGraph = graph.lock())
                    spGraph->parseNodeParamDependency(&sparam, param.defl);
            }
        }
    }
    for (const ParamPrimitive& param : dat.customUi.outputPrims)
    {
        auto iter = m_outputPrims.find(param.name);
        if (iter == m_outputPrims.end()) {
            add_output_prim_param(param);
            continue;
        }
        auto& sparam = iter->second;
        sparam.bVisible = param.bVisible;
        sparam.type = param.type;
    }
    for (const ParamObject& paramObj : dat.customUi.outputObjs)
    {
        add_output_obj_param(paramObj);
    }
}

ZENO_API bool INode::has_input(std::string const &id) const {
    //这个has_input在旧的语义里，代表的是input obj，如果有一些边没有连上，那么有一些参数值仅有默认值，未必会设这个input的，
    //还有一种情况，就是对象值是否有输入引入
    //这种情况要看旧版本怎么处理。
    //对于新版本而言，对于数值型输入，没有连上边仅有默认值，就不算has_input，有点奇怪，因此这里直接判断参数是否存在。
    auto iter = m_inputObjs.find(id);
    if (iter != m_inputObjs.end()) {
        return !iter->second.links.empty();
    }
    else {
        return m_inputPrims.find(id) != m_inputPrims.end();
    }
}

ZENO_API zany INode::get_input(std::string const &id) const {
    auto iter = m_inputPrims.find(id);
    if (iter != m_inputPrims.end()) {
        auto& val = iter->second.result;
        switch (iter->second.type) {
            case zeno::types::gParamType_Int:
            case zeno::types::gParamType_Float:
            case zeno::types::gParamType_Bool:
            case zeno::types::gParamType_Vec2f:
            case zeno::types::gParamType_Vec2i:
            case zeno::types::gParamType_Vec3f:
            case zeno::types::gParamType_Vec3i:
            case zeno::types::gParamType_Vec4f:
            case zeno::types::gParamType_Vec4i:
            {
                //依然有很多节点用了NumericObject，为了兼容，需要套一层NumericObject出去。
                std::shared_ptr<NumericObject> spNum = std::make_shared<NumericObject>();
                const auto& anyType = val.type();
                if (anyType == zeno::reflect::type_info<int>()) {
                    spNum->set<int>(zeno::reflect::any_cast<int>(val));
                }
                else if (anyType == zeno::reflect::type_info<bool>()) {
                    spNum->set<int>(zeno::reflect::any_cast<bool>(val));
                }
                else if (anyType == zeno::reflect::type_info<float>()) {
                    spNum->set<float>(zeno::reflect::any_cast<float>(val));
                }
                else if (anyType == zeno::reflect::type_info<vec2i>()) {
                    spNum->set<vec2i>(zeno::reflect::any_cast<vec2i>(val));
                }
                else if (anyType == zeno::reflect::type_info<vec3i>()) {
                    spNum->set<vec3i>(zeno::reflect::any_cast<vec3i>(val));
                }
                else if (anyType == zeno::reflect::type_info<vec4i>()) {
                    spNum->set<vec4i>(zeno::reflect::any_cast<vec4i>(val));
                }
                else if (anyType == zeno::reflect::type_info<vec2f>()) {
                    spNum->set<vec2f>(zeno::reflect::any_cast<vec2f>(val));
                }
                else if (anyType == zeno::reflect::type_info<vec3f>()) {
                    spNum->set<vec3f>(zeno::reflect::any_cast<vec3f>(val));
                }
                else if (anyType == zeno::reflect::type_info<vec4f>()) {
                    spNum->set<vec4f>(zeno::reflect::any_cast<vec4f>(val));
                }
                else
                {
                    //throw makeError<TypeError>(typeid(T));
                    //error, throw expection.
                }
                return spNum;
            }
            case zeno::types::gParamType_String:
            {
                const std::string& str = zeno::reflect::any_cast<std::string>(val);
                return std::make_shared<StringObject>(str);
                }
            default: {
            return nullptr;
                }
            }
        }
    else {
        auto iter2 = m_inputObjs.find(id);
        if (iter2 != m_inputObjs.end()) {
            return iter2->second.spObject;
        }
            return nullptr;
        }
}

ZENO_API void INode::set_pos(std::pair<float, float> pos) {
    m_pos = pos;
    CALLBACK_NOTIFY(set_pos, m_pos)
}

ZENO_API std::pair<float, float> INode::get_pos() const {
    return m_pos;
}

ZENO_API bool INode::in_asset_file() const {
    std::shared_ptr<Graph> spGraph = graph.lock();
    assert(spGraph);
    return getSession().assets->isAssetGraph(spGraph);
}

bool INode::set_primitive_input(std::string const& id, const zeno::reflect::Any& val) {
    auto iter = m_inputPrims.find(id);
    if (iter == m_inputPrims.end())
        return false;
    iter->second.result = val;
}

bool INode::set_primitive_output(std::string const& id, const zeno::reflect::Any& val) {
    auto iter = m_outputPrims.find(id);
    assert(iter != m_outputPrims.end());
    if (iter == m_outputPrims.end())
        return false;
    iter->second.result = val;
}

ZENO_API bool INode::set_output(std::string const& param, zany obj) {
    //只给旧节点模块使用，如果函数暴露reflect::Any，就会迫使所有使用这个函数的cpp文件include headers
    //会增加程序体积以及编译时间，待后续生成文件优化后再考虑处理。
    auto iter = m_outputObjs.find(param);
    if (iter != m_outputObjs.end()) {
            iter->second.spObject = obj;
        return true;
    }
    else {
        auto iter2 = m_outputPrims.find(param);
        if (iter2 != m_outputPrims.end()) {
            //兼容以前NumericObject的情况
            if (auto numObject = std::dynamic_pointer_cast<NumericObject>(obj)) {
                const auto& val = numObject->value;
                if (std::holds_alternative<int>(val))
                {
                    iter2->second.result = std::get<int>(val);
                }
                else if (std::holds_alternative<float>(val))
                {
                    iter2->second.result = std::get<float>(val);
                }
                else if (std::holds_alternative<vec2i>(val))
                {
                    iter2->second.result = std::get<vec2i>(val);
                }
                else if (std::holds_alternative<vec2f>(val))
                {
                    iter2->second.result = std::get<vec2f>(val);
                }
                else if (std::holds_alternative<vec3i>(val))
                {
                    iter2->second.result = std::get<vec3i>(val);
                }
                else if (std::holds_alternative<vec3f>(val))
                {
                    iter2->second.result = std::get<vec3f>(val);
                }
                else if (std::holds_alternative<vec4i>(val))
                {
                    iter2->second.result = std::get<vec4i>(val);
                }
                else if (std::holds_alternative<vec4f>(val))
                {
                    iter2->second.result = std::get<vec4f>(val);
                }
                else
                {
                    //throw makeError<TypeError>(typeid(T));
                    //error, throw expection.
                }
            }
            else if (auto strObject = std::dynamic_pointer_cast<StringObject>(obj)) {
                const auto& val = strObject->value;
                iter2->second.result = val;
            }
            return true;
        }
    }
    return false;
}

ZENO_API zany INode::get_output_obj(std::string const& param) {
    auto& spParam = safe_at(m_outputObjs, param, "miss output param `" + param + "` on node `" + m_name + "`");
    return spParam.spObject;
}

ZENO_API TempNodeCaller INode::temp_node(std::string const &id) {
    //TODO: deprecated
    std::shared_ptr<Graph> spGraph = graph.lock();
    assert(spGraph);
    return TempNodeCaller(spGraph.get(), id);
}

float INode::resolve(const std::string& expression, const ParamType type)
{
    std::string code = expression;
    Formula fmla(code, get_path());
    int ret = fmla.parse();
    if (ret == 0)
    {
        auto& funcMgr = zeno::getSession().funcManager;
        auto& astRoot = fmla.getASTResult();
        ZfxContext ctx;
        ctx.code = code;
        ctx.spNode = shared_from_this();
        zfxvariant res = funcMgr->calc(astRoot, &ctx);
        if (std::holds_alternative<int>(res)) {
            return std::get<int>(res);
        }
        else if (std::holds_alternative<float>(res)) {
            return std::get<float>(res);
        }
        else {
            throw makeError<UnimplError>();
        }
    }
    else {
        //TODO: kframe issues
    }
}

void INode::initTypeBase(zeno::reflect::TypeBase* pTypeBase)
{
    m_pTypebase = pTypeBase;
}

std::vector<std::pair<std::string, bool>> zeno::INode::getWildCardParams(const std::string& param_name, bool bPrim)
{
    std::vector<std::pair<std::string, bool>> params;
    if (bPrim)
    {
        std::string wildCardGroup;
        if (m_inputPrims.find(param_name) != m_inputPrims.end())
        {
            wildCardGroup = m_inputPrims.find(param_name)->second.wildCardGroup;
        }
        else if (m_outputPrims.find(param_name) != m_outputPrims.end())
        {
            wildCardGroup = m_outputPrims.find(param_name)->second.wildCardGroup;
        }
        for (const auto&[name, spParam] : m_inputPrims)
        {
            if (spParam.wildCardGroup == wildCardGroup)
            {
                if (!wildCardGroup.empty() || param_name == name) {
                    params.push_back({name, true});
            }
        }
        }
        for (const auto& [name, spParam] : m_outputPrims)
        {
            if (spParam.wildCardGroup == wildCardGroup)
            {
                if (!wildCardGroup.empty() || param_name == name) {
                    params.push_back({name, false});
            }
        }
        }
    } 
    else
    {
        std::string wildCardGroup;
        if (m_inputObjs.find(param_name) != m_inputObjs.end())
        {
            wildCardGroup = m_inputObjs.find(param_name)->second.wildCardGroup;
        }
        else if (m_outputObjs.find(param_name) != m_outputObjs.end())
        {
            wildCardGroup = m_outputObjs.find(param_name)->second.wildCardGroup;
        }
        for (const auto& [name, spParam] : m_inputObjs)
        {
            if (spParam.wildCardGroup == wildCardGroup)
            {
                if (!wildCardGroup.empty() || param_name == name) {
                    params.push_back({name, true});
            }
        }
        }
        for (const auto& [name, spParam] : m_outputObjs)
        {
            if (spParam.wildCardGroup == wildCardGroup)
            {
                if (!wildCardGroup.empty() || param_name == name) {
                    params.push_back({name, false});
            }
        }
    }
    }
    return params;
}

void zeno::INode::getParamTypeAndSocketType(const std::string& param_name, bool bPrim, bool bInput, ParamType& paramType, SocketType& socketType)
{
    if (bPrim) {
        auto iter = bInput ? m_inputPrims.find(param_name) : m_outputPrims.find(param_name);
        if (bInput ? (iter != m_inputPrims.end()) : (iter != m_outputPrims.end())) {
            paramType = iter->second.type;
            socketType = iter->second.socketType;
            return;
    }
    }
    else {
        auto iter = bInput ? m_inputObjs.find(param_name) : m_outputObjs.find(param_name);
        if (bInput ? (iter != m_inputObjs.end()) : (iter != m_outputObjs.end())) {
            paramType = iter->second.type;
            socketType = iter->second.socketType;
            return;
        }
    }
    paramType = Param_Null;
    socketType = Socket_Primitve;
}

template<class T, class E> T INode::resolveVec(const zeno::reflect::Any& defl, const ParamType type)
{
    if (zeno::reflect::get_type<E>() == defl.type()) {
        E vec = zeno::reflect::any_cast<E>(defl);
        T vecnum;
        for (int i = 0; i < vec.size(); i++) {
            float fVal = resolve(vec[i], type);
            vecnum[i] = fVal;
        }
        return vecnum;
    }
    else if (zeno::reflect::get_type<T>() == defl.type()) {
        return zeno::reflect::any_cast<T>(defl);
    }
    else {
        throw;
    }
}

}
