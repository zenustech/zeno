#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/Session.h>
#include <zeno/core/Assets.h>
#include <zeno/core/ObjectManager.h>
#include <zeno/core/INodeClass.h>
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
#include <zeno/core/FunctionManager.h>
#include "reflect/type.hpp"
#include <zeno/types/MeshObject.h>
#include "zeno_types/reflect/reflection.generated.hpp"
#include <zeno/core/reflectdef.h>
#include <zeno/formula/zfxexecute.h>
#include <zeno/extra/CalcContext.h>
#include <zeno/extra/SubnetNode.h>


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

ZENO_API INode::~INode() {
    int j;
    j = 0;
}

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
    zeno::CustomUI exportui = origin;

    for (auto& input_param : exportui.inputObjs) {
        std::string name = input_param.name;
        auto iterObj = m_inputObjs.find(name);
        assert(iterObj != m_inputObjs.end());
        input_param = iterObj->second.exportParam();
    }

    for (auto& tab : exportui.inputPrims) {
        for (auto& group : tab.groups) {
            for (auto& input_param : group.params) {
                std::string name = input_param.name;
                auto iterPrim = m_inputPrims.find(name);
                assert(iterPrim != m_inputPrims.end());
                input_param = iterPrim->second.exportParam();
            }
        }
    }

    for (auto& output_param : exportui.outputObjs) {
        std::string name = output_param.name;
        auto iterObj = m_outputObjs.find(name);
        assert(iterObj != m_outputObjs.end());
        output_param = iterObj->second.exportParam();
    }

    for (auto& output_param : exportui.outputPrims) {
        std::string name = output_param.name;
        auto iterPrim = m_outputPrims.find(name);
        assert(iterPrim != m_outputPrims.end());
        output_param = iterPrim->second.exportParam();
    }
    return exportui;


    exportui.nickname = origin.nickname;
    exportui.iconResPath = origin.iconResPath;
    exportui.doc = origin.doc;
    if (!origin.category.empty())
        exportui.category = origin.category;

    zeno::ParamGroup exportgroup;
    zeno::ParamTab exporttab;
    if (!origin.inputPrims.empty()) {
        exporttab.name = origin.inputPrims[0].name;
        if (!origin.inputPrims[0].groups.empty()) {
            exportgroup.name = origin.inputPrims[0].groups[0].name;
        }
    }
    for (const zeno::ParamTab& tab : origin.inputPrims) {
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
    exportui.inputPrims.emplace_back(std::move(exporttab));
    for (auto& [key, param] : m_inputPrims) {
        if (intputPrims.find(key) == intputPrims.end())
            exportui.inputPrims[0].groups[0].params.push_back(param.exportParam());
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

ZENO_API bool INode::isInDopnetwork()
{
    std::shared_ptr<Graph> parentGraph = graph.lock();
    while (parentGraph)
    {
        if (parentGraph->optParentSubgNode.has_value())
        {
            if (SubnetNode* subnet = parentGraph->optParentSubgNode.value()) {
                if (DopNetwork* dop = dynamic_cast<DopNetwork*>(subnet)) {
                    return true;
                }
                else {
                    parentGraph = subnet->getGraph().lock();
                }
            }
            else {
                break;
            }
        }
        else break;
    }
    return false;
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
        for (auto& [name, param] : m_inputPrims) {
            for (auto link : param.reflinks) {
                if (link->dest_inparam != &param) {
                    assert(link->dest_inparam);
                    auto destNode = link->dest_inparam->m_wpNode.lock();
                    destNode->mark_dirty(true);
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
                    inNode->mark_dirty(true);
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
                    inNode->mark_dirty(true);
                }
            }
        }
    }

    if (SubnetNode* pSubnetNode = dynamic_cast<SubnetNode*>(this))
    {
        if (bWholeSubnet)
            pSubnetNode->mark_subnetdirty(bOn);
        if (DopNetwork* pDop = dynamic_cast<DopNetwork*>(pSubnetNode)) {
            pDop->resetFrameState();
    }
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

void INode::preApply(CalcContext* pContext) {
    if (!m_dirty)
        return;

    reportStatus(true, Node_Pending);

    //TODO: the param order should be arranged by the descriptors.
    for (const auto& [name, param] : m_inputObjs) {
        bool ret = requireInput(name, pContext);
        if (!ret)
            zeno::log_warn("the param {} may not be initialized", name);
    }
    for (const auto& [name, param] : m_inputPrims) {
        bool ret = requireInput(name, pContext);
        if (!ret)
            zeno::log_warn("the param {} may not be initialized", name);
    }
}

void INode::preApplyTimeshift(CalcContext* pContext)
{
    int oldFrame = getSession().globalState->getFrameId();
    scope_exit sp([&oldFrame] { getSession().globalState->updateFrameId(oldFrame); });
    //get offset
    auto defl = get_input_prim_param("offset").defl;
    zeno::PrimVar offset = defl.has_value() ? zeno::reflect::any_cast<zeno::PrimVar>(defl) : 0;
    int newFrame = oldFrame + std::get<int>(offset);
    //clamp
    auto startFrameDefl = get_input_prim_param("start frame").defl;
    int globalStartFrame = getSession().globalState->getStartFrame();
    int startFrame = startFrameDefl.has_value() ? std::get<int>(zeno::reflect::any_cast<PrimVar>(startFrameDefl)) : globalStartFrame;
    auto endFrameDefl = get_input_prim_param("end frame").defl;
    int globalEndFrame = getSession().globalState->getEndFrame();
    int endFrame = endFrameDefl.has_value() ? std::get<int>(zeno::reflect::any_cast<PrimVar>(endFrameDefl)) : globalEndFrame;
    auto clampDefl = get_input_prim_param("clamp").defl;
    std::string clamp = clampDefl.has_value() ? zeno::reflect::any_cast<std::string>(clampDefl) : "None";
    if (startFrame > endFrame) {
        startFrame = globalStartFrame;
        endFrame = globalEndFrame;
    }
    if (clamp == "Clamp to First") {
        newFrame = newFrame < startFrame ? startFrame : newFrame;
    }
    else if (clamp == "Clamp to Last") {
        newFrame = newFrame > endFrame ? endFrame : newFrame;
    }
    else if (clamp == "Clamp to Both") {
        if (newFrame < startFrame) {
            newFrame = startFrame;
        }
        else if (newFrame > endFrame) {
            newFrame = endFrame;
        }
    }
    getSession().globalState->updateFrameId(newFrame);
    //propaget dirty
    propagateDirty(shared_from_this(), "$F");

    preApply(pContext);
}

void INode::reflectForeach_apply(CalcContext* pContext)
{
    std::string foreach_begin_path = zeno::reflect::any_cast<std::string>(get_defl_value("ForEachBegin Path"));
    if (std::shared_ptr<Graph> spGraph = graph.lock()) {
        auto foreach_begin = spGraph->getNode(foreach_begin_path);
        for (reset_forloop_settings(); is_continue_to_run(); increment())
        {
            foreach_begin->mark_dirty(true);

            preApply(pContext);
            reflectNode_apply();
        }
        auto output = get_output_obj("Output Object");
        output->update_key(m_uuid);
    }
}


ZENO_API void INode::apply() {

}

ZENO_API void INode::reflectNode_apply()
{
    if (m_pTypebase) {
        for (zeno::reflect::IMemberFunction* func : m_pTypebase->get_member_functions()) {
            const auto& funcname = func->get_name();
            if (funcname == "apply") {
                //根据ReflectCustomUI获取fieldName到displayName映射
                std::map<std::string, std::string> mappingInputParams;
                std::vector<std::string> mappingReturnParams;

                zeno::_ObjectParam retInfoOnReflectUI;
                getNameMappingFromReflectUI(m_pTypebase, shared_from_this(), mappingInputParams, mappingReturnParams);

                //从apply参数获取输入
                zeno::reflect::ArrayList<zeno::reflect::Any> paramValues;
                std::vector<std::tuple<std::string, zeno::ParamType, int>> outputsName;

                const zeno::reflect::ArrayList<zeno::reflect::RTTITypeInfo>& params = func->get_params();
                const auto& field_names = func->get_params_name();
                for (int i = 0; i < params.size(); i++) {
                    const zeno::reflect::RTTITypeInfo& param_type = params[i];
                    const std::string field_name(field_names[i].c_str());
                    std::string normal_name = field_name;
                    auto iterMapping = mappingInputParams.find(field_name);
                    if (iterMapping != mappingInputParams.end()) {
                        normal_name = iterMapping->second;
                    }

                    zeno::reflect::Any inputAny;
                    bool bConstPtr = false;
                    zeno::isObjectType(param_type, bConstPtr);

                    auto iter = m_inputPrims.find(normal_name);
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
                        auto iter2 = m_inputObjs.find(normal_name);
                        if (iter2 != m_inputObjs.end()) {
                            inputAny = iter2->second.spObject;
                        }
                    }
                    paramValues.add_item(inputAny);
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

                const zeno::reflect::RTTITypeInfo& ret_rtti = func->get_return_rtti();
                ParamType _type = ret_rtti.get_decayed_hash() == 0 ? ret_rtti.hash_code() : ret_rtti.get_decayed_hash();
                bool bConstPtr = false;

                auto funcSetOutputParam = [&](const std::string& normalName, const zeno::reflect::Any& returnVal) {
                    auto iterOutputObj = m_outputObjs.find(normalName);
                    if (iterOutputObj != m_outputObjs.end()) {
                        iterOutputObj->second.spObject = any_cast<zany>(returnVal);
                    }
                    else {
                        auto iterOutputPrim = m_outputPrims.find(normalName);
                        if (iterOutputPrim != m_outputPrims.end()) {
                            iterOutputPrim->second.result = returnVal;
                        }
                    }
                };

                if (ret_rtti.flags() & TF_IsMultiReturn) {
                    ArrayList<RTTITypeInfo> rets = func->get_multi_return_rtti();
                    
                    std::vector<Any> retVec = any_cast<std::vector<Any>>(res);
                    //TODO: 有一种可能，就是映射的名称只有一部分，会导致大小和位置不好对的上，后续看看怎么处理
                    assert(rets.size() == mappingReturnParams.size() && retVec.size() == rets.size());
                    for (int i = 0; i < mappingReturnParams.size(); i++) {
                        funcSetOutputParam(mappingReturnParams[i], retVec[i]);
                    }
                }
                else if (!mappingReturnParams.empty()){
                    funcSetOutputParam(mappingReturnParams[0], res);
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

void INode::registerObjToManager()
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

void INode::on_link_added_removed(bool bInput, const std::string& paramname, bool bAdded) {
    checkParamsConstrain();
}

void INode::on_node_about_to_remove() {
    //移除所有引用边的依赖关系
    for (auto& [_, input_param] : m_inputPrims)
    {
        for (const std::shared_ptr<ReferLink>& reflink : input_param.reflinks) {
            if (reflink->source_inparam == &input_param) {
                //当前参数是引用源
                auto& otherLinks = reflink->dest_inparam->reflinks;
                otherLinks.erase(std::remove(otherLinks.begin(), otherLinks.end(), reflink));
                //参数值也改掉吧，把ref(...)改为 inv_ref(...)
                auto otherNode = reflink->dest_inparam->m_wpNode.lock();
                assert(otherNode);

                auto defl = reflink->dest_inparam->defl;
                assert(defl.has_value());
                ParamType type = defl.type().hash_code();
                if (type == gParamType_PrimVariant) {
                    PrimVar& var = any_cast<PrimVar>(defl);
                    std::visit([&](auto& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, std::string>) {
                            arg.replace(arg.find("ref("), 4, "ref_not_exist(");
                        }
                    }, var);
                    otherNode->update_param(reflink->dest_inparam->name, var);
                }
                else if (type == gParamType_VecEdit) {
                    vecvar vec = any_cast<vecvar>(defl);
                    for (auto& elem : vec) {
                        std::visit([&](auto& arg) {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr (std::is_same_v<T, std::string>) {
                                arg.replace(arg.find("ref("), 4, "ref_not_exist(");
                            }
                        }, elem);
                    }
                    otherNode->update_param(reflink->dest_inparam->name, vec);
                }

                if (otherNode)
                    otherNode->mark_dirty(true);
            }
            else {
                //当前参数引用了别的节点参数
                auto& otherLinks = reflink->source_inparam->reflinks;
                otherLinks.erase(std::remove(otherLinks.begin(), otherLinks.end(), reflink));
                auto otherNode = reflink->source_inparam->m_wpNode.lock();
                if (otherNode)
                    otherNode->mark_dirty(true);
            }
        }
        input_param.reflinks.clear();
    }
}

void INode::onNodeNameUpdated(const std::string& oldname, const std::string& newname) {
    std::string graphpath = get_graph_path();
    std::string oldpath = graphpath + '/' + oldname;
    std::string newpath = graphpath + '/' + newname;

    //检查所有reflink，将目标参数的引用名称调整一下
    for (const auto& [_, param] : m_inputPrims) {
        for (auto reflink : param.reflinks) {
            if (reflink->dest_inparam != &param) {
                //直接修改dest_inparam->defl.
                bool bUpdate = false;

                auto fUpdateParamDefl = [oldpath, newpath, graphpath, &bUpdate](std::string& arg) {
                    auto matchs = zeno::getReferPath(arg);
                    for (const auto& str : matchs)
                    {
                        std::string absolutePath = zeno::absolutePath(graphpath, str);
                        if (absolutePath.find(oldpath) != std::string::npos)
                        {
                            std::regex num_rgx("[0-9]+");
                            //如果是数字，需要将整个refer替换
                            if (std::regex_match(newpath, num_rgx))
                            {
                                arg = newpath;
                                bUpdate = true;
                                break;
                            }
                            else
                            {
                                std::regex pattern(oldpath);
                                std::string format = regex_replace(absolutePath, pattern, newpath);
                                //relative path
                                if (absolutePath != str)
                                {
                                    format = zeno::relativePath(graphpath, format);
                                }
                                std::regex rgx(str);
                                arg = regex_replace(arg, rgx, format);
                            }
                            bUpdate = true;
                        }
                    }
                };

                zeno::reflect::Any adjustParamVal = reflink->dest_inparam->defl;

                assert(adjustParamVal.has_value());
                ParamType type = adjustParamVal.type().hash_code();
                if (type == zeno::types::gParamType_PrimVariant) {
                    PrimVar& var = zeno::reflect::any_cast<PrimVar>(adjustParamVal);
                    std::visit([&](auto& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, std::string>) {
                            fUpdateParamDefl(arg);
                        }
                        else {
                            assert(false);
                            zeno::log_warn("error param type");
                        }
                        }, var);
                    if (bUpdate) {
                        adjustParamVal = zeno::reflect::move(var);
                    }
                }
                else if (type == zeno::types::gParamType_VecEdit) {
                    vecvar var = zeno::reflect::any_cast<vecvar>(adjustParamVal);
                    for (PrimVar& elem : var)
                    {
                        std::visit([&](auto& arg) {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr (std::is_same_v<T, std::string>) {
                                fUpdateParamDefl(arg);
                            }
                            }, elem);
                    }
                    if (bUpdate) {
                        adjustParamVal = zeno::reflect::move(var);
                    }
                }
                else {
                    assert(false);
                    zeno::log_error("unknown param type of refer param");
                }

                if (bUpdate) {
                    auto spDestNode = reflink->dest_inparam->m_wpNode.lock();
                    spDestNode->update_param(reflink->dest_inparam->name, adjustParamVal);
                }
            }
        }
    }
}

void INode::constructReference(const std::string& param_name) {
    auto iter = m_inputPrims.find(param_name);
    if (iter == m_inputPrims.end())
        return;

    const Any& param_defl = iter->second.defl;
    initReferLinks(&iter->second);
}

void INode::initReferLinks(PrimitiveParam* target_param) {
    std::set<std::pair<std::string, std::string>> refSources = resolveReferSource(target_param->defl);
    auto newAdded = refSources;

    for (auto iter = target_param->reflinks.begin(); iter != target_param->reflinks.end(); )
    {
        bool bExist = false;
        std::shared_ptr<ReferLink> spRefLink = (*iter);
        PrimitiveParam* remote_source = spRefLink->source_inparam;
        assert(remote_source);
        if (remote_source == target_param) {
            iter++;
            continue;
        }

        //查看当前link在新的集合里是否还存在。
        for (const auto& [source_node_uuidpath, source_param] : refSources)
        {
            auto spSrcNode = remote_source->m_wpNode.lock();
            if (spSrcNode->get_uuid_path() == source_node_uuidpath &&
                remote_source->name == source_param) {
                //已经有了
                bExist = true;
                newAdded.erase({ source_node_uuidpath, source_param });
                break;
            }
        }

        if (bExist) {
            iter++;
        }
        else {
            iter = target_param->reflinks.erase(iter);
            auto& other_links = remote_source->reflinks;
            other_links.erase(std::remove(other_links.begin(), other_links.end(), spRefLink));
        }
    }

    for (const auto& [source_node_uuidpath, source_param] : newAdded)
    {
        std::shared_ptr<INode> srcNode = getSession().mainGraph->getNodeByUuidPath(source_node_uuidpath);
        auto iterSrcParam = srcNode->m_inputPrims.find(source_param);
        if (iterSrcParam != srcNode->m_inputPrims.end()) {
            PrimitiveParam& srcparam = iterSrcParam->second;
            if (&srcparam != target_param)  //排除直接引用自己的情况
            {
                //构造reflink
                std::shared_ptr<ReferLink> reflink = std::make_shared<ReferLink>();
                reflink->source_inparam = &srcparam;
                reflink->dest_inparam = target_param;
                target_param->reflinks.push_back(reflink);
                srcparam.reflinks.push_back(reflink);
            }
        }
    }
}

std::set<std::pair<std::string, std::string>> INode::resolveReferSource(const Any& param_defl) {

    std::set<std::pair<std::string, std::string>> refSources;
    std::vector<std::string> refSegments;

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

std::shared_ptr<DictObject> INode::processDict(ObjectParam* in_param, CalcContext* pContext) {
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
                    outNode->doApply(pContext);
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
                    outNode->doApply(pContext);
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

std::shared_ptr<ListObject> INode::processList(ObjectParam* in_param, CalcContext* pContext) {
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
                    outNode->doApply(pContext);
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
                    outNode->doApply(pContext);
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
    if (!in_param || in_param->type == Param_Wildcard) {
        return nullptr;
    }

    int frame = getGlobalState()->getFrameId();

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
                    float res = resolve(arg, type);
                    return (type == gParamType_Int) ? zeno::reflect::make_any<int>(res) : 
                        zeno::reflect::make_any<float>(res);
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

bool INode::receiveOutputObj(ObjectParam* in_param, std::shared_ptr<INode> outNode, zany outputObj, ParamType outobj_type) {

    if (in_param->socketType == Socket_Clone) {
        in_param->spObject = outputObj->clone();
    }
    else if (in_param->socketType == Socket_Owning) {
        in_param->spObject = outputObj->move_clone();
        assert(outNode);
        outNode->mark_dirty(true);
    }
    else if (in_param->socketType == Socket_ReadOnly) {
        in_param->spObject = outputObj;
        //TODO: readonly property on object.
    }
    else if (in_param->socketType == Socket_WildCard) {
        if (std::shared_ptr<zeno::INode> node = in_param->m_wpNode.lock()) {
            if (node->get_nodecls() == "SubOutput") {
                in_param->spObject = outputObj;
            }
        }
    }
    return true;
}

ZENO_API bool INode::requireInput(std::string const& ds, CalcContext* pContext) {
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
                    std::shared_ptr<DictObject> outDict = processDict(in_param, pContext);
                    receiveOutputObj(in_param, nullptr, outDict, gParamType_Dict);
                    break;
                }
                case gParamType_List:
                {
                    std::shared_ptr<ListObject> outList = processList(in_param, pContext);
                    receiveOutputObj(in_param, nullptr, outList, gParamType_List);
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
                            outNode->doApply(pContext);
                        }, outNode.get());

                        if (out_param->spObject)
                        {
                            receiveOutputObj(in_param, outNode, out_param->spObject, out_param->type);
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

                std::list<std::shared_ptr<ReferLink>> depRefLinks;
                for (auto reflink : in_param->reflinks) {
                    if (reflink->source_inparam != in_param) {
                        depRefLinks.push_back(reflink);
                    }
                }

                if (!depRefLinks.empty()) {
                    for (auto reflink : depRefLinks) {
                        assert(reflink->source_inparam);
                        auto spSrcNode = reflink->source_inparam->m_wpNode.lock();
                        assert(spSrcNode);
                        std::shared_ptr<Graph> spSrcGraph = spSrcNode->graph.lock();
                        assert(spSrcGraph);
                        spSrcNode->doApply_Parameter(reflink->source_inparam->name, pContext);
                    }
                }

                in_param->result = processPrimitive(in_param);
                //旧版本的requireInput指的是是否有连线，如果想兼容旧版本，这里可以返回false，但使用量不多，所以就修改它的定义。
            }
            else {
                if (in_param->links.size() == 1) {
                    std::shared_ptr<PrimitiveLink> spLink = *in_param->links.begin();
                    std::shared_ptr<INode> outNode = spLink->fromparam->m_wpNode.lock();

                    GraphException::translated([&] {
                        outNode->doApply(pContext);
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

void INode::doApply_Parameter(std::string const& name, CalcContext* pContext) {
    if (!m_dirty) {
        return;
    }

    std::string uuid_path = get_uuid_path() + "/" + name;
    if (pContext->uuid_node_params.find(uuid_path) != pContext->uuid_node_params.end()) {
        throw makeError<UnimplError>("cycle reference occurs when refer paramters!");
    }

    scope_exit scope_apply_param([&]() { pContext->uuid_node_params.erase(uuid_path); });
    pContext->uuid_node_params.insert(uuid_path);

    requireInput(name, pContext);
}

ZENO_API void INode::doApply(CalcContext* pContext) {

    if (!m_dirty) {
        registerObjToManager();//如果只是打view，也是需要加到manager的。
        return;
    }

    assert(pContext);
    std::string uuid_path = get_uuid_path();
    if (pContext->visited_nodes.find(uuid_path) != pContext->visited_nodes.end()) {
        throw makeError<UnimplError>("cycle reference occurs!");
    }
    pContext->visited_nodes.insert(uuid_path);
    scope_exit spUuidRecord([=] {pContext->visited_nodes.erase(uuid_path); });

    if (m_nodecls == "TimeShift") {
        preApplyTimeshift(pContext);
    } else if (m_nodecls == "ForEachEnd") {
    } else {
        preApply(pContext);
    }

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
            if (m_nodecls == "ForEachEnd") {
                reflectForeach_apply(pContext);
            } else {
                reflectNode_apply();
            }
        }
    }
    log_debug("==> leave {}", m_name);

    //DopNetwork
    if (DopNetwork* dop = dynamic_cast<DopNetwork*>(this)) {
        reportStatus(true, Node_Running);
        registerObjToManager();
        reportStatus(true, Node_DirtyReadyToRun);
    } else {
        if (m_nodecls == "ForEachEnd") {
            reportStatus(true, Node_Running);
            registerObjToManager();
        } else {
            registerObjToManager();
            reportStatus(false, Node_RunSucceed);
        }
    }
}

ZENO_API CommonParam INode::get_input_param(std::string const& name, bool* bExist) {
    auto primparam = get_input_prim_param(name, bExist);
    if (bExist && *bExist)
        return primparam;
    auto objparam = get_input_obj_param(name, bExist);
    if (bExist && *bExist)
        return objparam;
    if (bExist)
        *bExist = false;
}

ZENO_API CommonParam INode::get_output_param(std::string const& name, bool* bExist) {
    auto primparam = get_output_prim_param(name, bExist);
    if (bExist && *bExist)
        return primparam;
    auto objparam = get_output_obj_param(name, bExist);
    if (bExist && *bExist)
        return objparam;
    if (bExist)
        *bExist = false;
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
        param.bSocketVisible = spParamObj.bSocketVisible;
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

ZENO_API zeno::reflect::Any INode::get_defl_value(std::string const& name) {
    //向量情况也挺麻烦的，因为可能存在公式
    ParamPrimitive param;
    auto iter = m_inputPrims.find(name);
    if (iter != m_inputPrims.end()) {
        zeno::reflect::Any defl = iter->second.defl;
        //不支持取公式，因为公式要引发计算，很麻烦
        convertToOriginalVar(defl, iter->second.type);
        return defl;
    }else{
        return zeno::reflect::Any();
    }
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
    sparam.bSocketVisible = param.bSocketVisible;
    sparam.wildCardGroup = param.wildCardGroup;
    sparam.sockprop = param.sockProp;
    //sparam.bInnerParam = param.bInnerParam;
    sparam.constrain = param.constrain;
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
    sparam.constrain = param.constrain;
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
    sparam.bSocketVisible = param.bSocketVisible;
    sparam.constrain = param.constrain;
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
    sparam.constrain = param.constrain;
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
        if (!node.customUi.inputPrims.empty() && !node.customUi.inputPrims[0].groups.empty()) {
            for (auto& [name, paramPrimitive] : m_inputPrims) {
                node.customUi.inputPrims[0].groups[0].params.push_back(paramPrimitive.exportParam());
            }
        }
    }
    else {
    for (auto &tab : node.customUi.inputPrims)
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
    zeno::reflect::Any old_value;
    bool ret = update_param_impl(param, new_value, old_value);
    if (ret) {
        CALLBACK_NOTIFY(update_param, param, old_value, new_value)
        mark_dirty(true);
    }
    return ret;
}

ZENO_API bool INode::update_param_impl(const std::string& param, zeno::reflect::Any new_value, zeno::reflect::Any& old_value)
{
    auto& spParam = safe_at(m_inputPrims, param, "miss input param `" + param + "` on node `" + m_name + "`");
    bool isvalid = convertToEditVar(new_value, spParam.type);
    if (!isvalid) {
        zeno::log_error("cannot convert to edit variable");
        return false;
    }
    if (spParam.defl != new_value)
    {
        old_value = spParam.defl;
        spParam.defl = new_value;

        std::shared_ptr<Graph> spGraph = graph.lock();
        assert(spGraph);

        spGraph->onNodeParamUpdated(&spParam, old_value, new_value);
        initReferLinks(&spParam);
        checkParamsConstrain();
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
                    CALLBACK_NOTIFY(update_param_type, param, type, bInput)

                    //默认值也要更新
                    if (bInput) {
                        zeno::reflect::Any defl = initAnyDeflValue(type);
                        convertToEditVar(defl, type);
                        update_param(spParam.name, defl);
                    }
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
                    CALLBACK_NOTIFY(update_param_type, param, type, bInput)
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

ZENO_API bool INode::update_param_visible(const std::string& name, bool bOn, bool bInput) {
    if (bInput) {
        if (auto iter = m_inputObjs.find(name); iter != m_inputObjs.end()) {
            auto& paramObj = iter->second;
            if (paramObj.bVisible != bOn) {
                paramObj.bVisible = bOn;
                return true;
            }
        }
        else if (auto iter = m_inputPrims.find(name); iter != m_inputPrims.end()){
            auto& paramPrim = iter->second;
            if (paramPrim.bVisible != bOn) {
                paramPrim.bVisible = bOn;
                return true;
            }
        }
    }
    else {
        if (auto iter = m_outputObjs.find(name); iter != m_outputObjs.end()) {
            auto& paramObj = iter->second;
            if (paramObj.bVisible != bOn) {
                paramObj.bVisible = bOn;
                return true;
            }
        }
        else if (auto iter = m_outputPrims.find(name); iter != m_outputPrims.end()) {
            auto& paramPrim = iter->second;
            if (paramPrim.bVisible != bOn) {
                paramPrim.bVisible = bOn;
                return true;
            }
        }
    }
    return false;
}

void INode::checkParamsConstrain() {
    //ZfxContext
    auto& funcMgr = zeno::getSession().funcManager;
    ZfxContext ctx;
    ctx.spNode = shared_from_this();
    //对于所有带有约束的输入参数，调整其可见和可用情况

    std::set<std::string> adjInputs, adjOutputs;

    bool bParamPropChanged = false;
    for (const auto& [name, param] : m_inputObjs) {
        if (!param.constrain.empty()) {
            ctx.code = param.constrain;
            ctx.param_constrain.constrain_param = name;
            ctx.param_constrain.bInput = true;
            ZfxExecute zfx(ctx.code, &ctx);
            zfx.execute();
            if (ctx.param_constrain.update_nodeparam_prop) {
                bParamPropChanged = true;
                adjInputs.insert(name);
            }
        }
    }
    for (const auto& [name, param] : m_inputPrims) {
        if (!param.constrain.empty()) {
            ctx.code = param.constrain;
            ctx.param_constrain.constrain_param = name;
            ctx.param_constrain.bInput = true;
            ZfxExecute zfx(ctx.code, &ctx);
            zfx.execute();
            if (ctx.param_constrain.update_nodeparam_prop) {
                bParamPropChanged = true;
                adjInputs.insert(name);
            }
        }
    }
    for (const auto& [name, param] : m_outputPrims) {
        if (!param.constrain.empty()) {
            ctx.code = param.constrain;
            ctx.param_constrain.constrain_param = name;
            ctx.param_constrain.bInput = false;
            ZfxExecute zfx(ctx.code, &ctx);
            zfx.execute();
            if (ctx.param_constrain.update_nodeparam_prop) {
                bParamPropChanged = true;
                adjOutputs.insert(name);
            }
        }
    }
    for (const auto& [name, param] : m_outputObjs) {
        if (!param.constrain.empty()) {
            ctx.code = param.constrain;
            ctx.param_constrain.constrain_param = name;
            ctx.param_constrain.bInput = false;
            ZfxExecute zfx(ctx.code, &ctx);
            zfx.execute();
            if (ctx.param_constrain.update_nodeparam_prop) {
                bParamPropChanged = true;
                adjOutputs.insert(name);
            }
        }
    }

    if (bParamPropChanged) {
        //通知上层UI去统一更新
        CALLBACK_NOTIFY(update_visable_enable, this, adjInputs, adjOutputs)
    }
}

ZENO_API bool INode::update_param_enable(const std::string& name, bool bOn, bool bInput) {
    if (bInput) {
        if (auto iter = m_inputObjs.find(name); iter != m_inputObjs.end()) {
            auto& paramObj = iter->second;
            if (paramObj.bEnable != bOn) {
                paramObj.bEnable = bOn;
                return true;
            }
        }
        else if (auto iter = m_inputPrims.find(name); iter != m_inputPrims.end()) {
            auto& paramPrim = iter->second;
            if (paramPrim.bEnable != bOn) {
                paramPrim.bEnable = bOn;
                return true;
            }
        }
        else {
            return false;
        }
    }
    else {
        if (auto iter = m_outputObjs.find(name); iter != m_outputObjs.end()) {
            auto& paramObj = iter->second;
            if (paramObj.bEnable != bOn) {
                paramObj.bEnable = bOn;
                return true;
            }
        }
        else if (auto iter = m_outputPrims.find(name); iter != m_outputPrims.end()) {
            auto& paramPrim = iter->second;
            if (paramPrim.bEnable != bOn) {
                paramPrim.bEnable = bOn;
                return true;
            }
        }
        else {
            return false;
        }
    }
    return false;
}

ZENO_API bool zeno::INode::update_param_socket_visible(const std::string& param, bool bVisible, bool bInput)
{
    CORE_API_BATCH
    if (bInput) {
        auto& spParam = safe_at(m_inputPrims, param, "miss input param `" + param + "` on node `" + m_name + "`");
        if (spParam.bSocketVisible != bVisible)
        {
            spParam.bSocketVisible = bVisible;
            CALLBACK_NOTIFY(update_param_socket_visible, param, bVisible, bInput)
            return true;
        }
    }
    else {
        auto& spParam = safe_at(m_outputPrims, param, "miss output param `" + param + "` on node `" + m_name + "`");
        if (spParam.bSocketVisible != bVisible)
        {
            spParam.bSocketVisible = bVisible;
            CALLBACK_NOTIFY(update_param_socket_visible, param, bVisible, bInput)
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
                sparam.bSocketVisible = param.bSocketVisible;
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
    for (auto tab : dat.customUi.inputPrims)
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
                convertToEditVar(param.defl, param.type);
                sparam.defl = param.defl;
                convertToEditVar(sparam.defl, param.type);

                // 普通子图的控件及参数类型，是由定义处决定的，而非IO值。
                //sparam.control = param.control;
                //sparam.ctrlProps = param.ctrlProps;
                //sparam.type = param.type;
                sparam.bSocketVisible = param.bSocketVisible;

                //graph记录$F相关节点
                if (std::shared_ptr<Graph> spGraph = graph.lock())
                    spGraph->parseNodeParamDependency(&sparam, sparam.defl);
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
        sparam.bSocketVisible = param.bSocketVisible;
        //sparam.type = param.type;
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

ZENO_API bool INode::is_continue_to_run() {
    return false;
}

ZENO_API void INode::increment() {

}

ZENO_API void INode::reset_forloop_settings() {

}

ZENO_API std::shared_ptr<IObject> INode::get_iterate_object() {
    return nullptr;
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
