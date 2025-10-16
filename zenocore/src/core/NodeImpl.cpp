#include <zeno/core/NodeImpl.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/Session.h>
#include <zeno/core/Assets.h>
#include <zeno/core/INodeClass.h>
#include <zeno/types/DummyObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/extra/foreach.h>
#include <zeno/utils/Error.h>
#include <zeno/utils/string.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#ifdef ZENO_BENCHMARKING
#include <zeno/utils/Timer.h>
#endif
#include <zeno/core/IObject.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/uuid.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/CoreParam.h>
#include <zeno/types/DictObject.h>
#include <zeno/ListObject.h>
#include <zeno/utils/helper.h>
#include <zeno/utils/uuid.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/GraphException.h>
#include <zeno/formula/formula.h>
#include <zeno/core/FunctionManager.h>
#include <reflect/type.hpp>
#include <reflect/container/arraylist>
#include <reflect/metadata.hpp>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/MeshObject.h>
#include <zeno/types/MatrixObject.h>
#include "zeno_types/reflect/reflection.generated.hpp"
#include <zeno/core/reflectdef.h>
#include <zeno/formula/zfxexecute.h>
#include <zeno/extra/CalcContext.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/utils/interfaceutil.h>
//#include <Python.h>
//#include <pybind11/pybind11.h>


using namespace zeno::reflect;
using namespace zeno::types;


namespace zeno {

    class ForEachEnd;

NodeImpl::NodeImpl(INode* pNode) : m_pNode(pNode), m_pGraph(nullptr) {
    if (m_pNode)
        m_pNode->m_pAdapter = this;
}

NodeType NodeImpl::nodeType() const {
    return m_pNode->type();
}

bool NodeImpl::is_locked() const {
    return false;
}

void NodeImpl::set_locked(bool bLocked) {

}

void NodeImpl::initUuid(Graph* pGraph, const std::string nodecls) {
    //TODO: 考虑asset的情况
    m_nodecls = nodecls;
    this->m_pGraph = pGraph;

    m_uuid = generateUUID(nodecls);
    ObjPath path;
    path += m_uuid;
    while (pGraph) {
        const std::string name = pGraph->getName();
        if (name == "main") {
            break;
        }
        else {
            if (!pGraph->getParentSubnetNode())
                break;
            auto pSubnetNode = pGraph->getParentSubnetNode();
            assert(pSubnetNode);
            path = (pSubnetNode->m_uuid) + "/" + path;
            pGraph = pSubnetNode->m_pGraph;
        }
    }
    m_uuidPath = path;
}

NodeImpl::~NodeImpl() {
    int j;
    j = 0;
}

Graph* NodeImpl::getThisGraph() const {
    return m_pGraph;
}

Session *NodeImpl::getThisSession() const {
    return &getSession();
}

GlobalState *NodeImpl::getGlobalState() const {
    return getSession().globalState.get();
}

std::string NodeImpl::get_nodecls() const
{
    return m_nodecls;
}

std::string NodeImpl::get_ident() const
{
    return m_name;
}

std::string NodeImpl::get_show_name() const {
    if (nodeClass) {
        std::string dispName = nodeClass->m_customui.nickname;
        if (!dispName.empty())
            return dispName;
    }
    return m_nodecls;
}

std::string NodeImpl::get_show_icon() const {
    if (nodeClass) {
        return nodeClass->m_customui.uistyle.iconResPath;
    }
    else {
        return "";
    }
}

ObjPath NodeImpl::get_uuid_path() const {
    return m_uuidPath;
}

CustomUI NodeImpl::get_customui() const {
    if (nodeClass) {
        return nodeClass->m_customui;
    }
    else {
        return CustomUI();
    }
}

Graph* NodeImpl::getGraph() const {
    return m_pGraph;
}

INode* NodeImpl::coreNode() const {
    return m_pNode.get();
}

ObjPath NodeImpl::get_graph_path() const {
    ObjPath path;
    path = "";

    Graph* pGraph = m_pGraph;

    while (pGraph) {
        const std::string name = pGraph->getName();
        if (name == "main") {
            path = "/main/" + path;
            break;
        }
        else {
            if (!pGraph->getParentSubnetNode())
                break;
            auto pSubnetNode = pGraph->getParentSubnetNode();
            assert(pSubnetNode);
            path = pSubnetNode->m_name + "/" + path;
            pGraph = pSubnetNode->m_pGraph;
        }
    }
    return path;
}

CustomUI NodeImpl::_deflCustomUI() const {
    if (!nodeClass)
        return CustomUI();

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
}

CustomUI NodeImpl::export_customui() const
{
    CustomUI deflUI = _deflCustomUI();
    INode* pNode = coreNode();
    if (pNode) {
        CustomUI nodeui = pNode->export_customui();
        if (nodeui.inputObjs.empty() && nodeui.inputPrims.empty() &&
            nodeui.outputPrims.empty() && nodeui.outputObjs.empty()) {
            if (nodeui.uistyle.background != "" || nodeui.uistyle.iconResPath != "") {
                deflUI.uistyle = nodeui.uistyle;
            }
            return deflUI;
        }
        else {
            return nodeui;
        }
    }
    return deflUI;

#if 0
    exportui.nickname = origin.nickname;
    exportui.uistyle = origin.uistyle;
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
#endif
}

ObjPath NodeImpl::get_path() const {
    ObjPath path;
    path = m_name;

    Graph* pGraph = m_pGraph;

    while (pGraph) {
        const std::string name = pGraph->getName();
        if (name == "main") {
            path = "/main/" + path;
            break;
        }
        else {
            path = name + "/" + path;
            if (!pGraph->getParentSubnetNode())
                break;
            auto pSubnetNode = pGraph->getParentSubnetNode();
            assert(pSubnetNode);
            pGraph = pSubnetNode->m_pGraph;
        }
    }
    return path;
}

std::string NodeImpl::get_uuid() const
{
    return m_uuid;
}

std::string NodeImpl::get_name() const
{
    return m_name;
}

void NodeImpl::set_name(const std::string& customname)
{
    m_name = customname;
}

void NodeImpl::set_view(bool bOn)
{
    //这一类方法要和ui model端位于同一线程
    if (m_bView == bOn) {
        return;
    }
    {
        CORE_API_BATCH

        m_bView = bOn;
        CALLBACK_NOTIFY(set_view, m_bView)

        Graph* spGraph = m_pGraph;
        assert(spGraph);
        spGraph->viewNodeUpdated(m_name, bOn);
    }
}

bool NodeImpl::is_view() const
{
    return m_bView;
}

void NodeImpl::set_bypass(bool bOn)
{
    CORE_API_BATCH

    m_bypass = bOn;
    CALLBACK_NOTIFY(set_bypass, m_bypass)
    mark_dirty(true);

    //Graph* spGraph = graph;
    //assert(spGraph);
    //spGraph->viewNodeUpdated(m_name, bOn);
}

bool NodeImpl::is_bypass() const
{
    return m_bypass;
}

void NodeImpl::set_nocache(bool bOn) {
    CORE_API_BATCH

    m_nocache = bOn;
    CALLBACK_NOTIFY(set_nocache, m_nocache)
    mark_dirty(true);
}

bool NodeImpl::is_nocache() const {
    return m_nocache;
}

void NodeImpl::reportStatus(bool bDirty, NodeRunStatus status) {
    m_status = status;
    m_dirty = bDirty;
    auto& sess = zeno::getSession();
    if (status == Node_RunSucceed) {
        sess.globalState->update_consume_time(this->time());
    }
    sess.reportNodeStatus(m_uuidPath, bDirty, status);
}

void NodeImpl::mark_previous_ref_dirty() {
    mark_dirty(true);
    //不仅要自身标脏，如果前面的节点是以引用的方式连接，说明前面的节点都可能被污染了，所有都要标脏。
    //TODO: 由端口而不是边控制。
    /*
    for (const auto& [name, param] : m_inputs) {
        for (const auto& link : param.links) {
            if (link->lnkProp == Link_Ref) {
                auto spOutParam = link->fromparam.lock();
                auto spPreviusNode = spOutParam->m_wpNode;
                spPreviusNode->mark_previous_ref_dirty();
            }
        }
    }
    */
}

bool NodeImpl::only_single_output_object() const {
    return m_outputObjs.size() == 1 && m_outputPrims.empty();
}

bool NodeImpl::has_frame_relative_params() const {
    for (auto& [name, param] : m_inputPrims) {
        assert(param.defl.has_value());
        const std::string& uuid = get_uuid();
        if (gParamType_String == param.type) {
            if (param.defl.type().hash_code() == gParamType_PrimVariant) {//type是string，实际defl可能是primvar
                const zeno::PrimVar& editVar = zeno::reflect::any_cast<zeno::PrimVar>(param.defl);
                return std::visit([](auto&& arg)-> bool {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::string>) {
                        std::string defl = zeno::reflect::any_cast<std::string>(arg);
                        if (defl.find("$F") != std::string::npos)
                            return true;
                    }
                    return false;
                }, editVar);
            } else if (param.defl.type().hash_code() == gParamType_String) {
                std::string defl = zeno::any_cast_to_string(param.defl);
                if (defl.find("$F") != std::string::npos) {
                    return true;
                }
            }
        }
        else if (gParamType_Int == param.type || gParamType_Float == param.type || gParamType_PrimVariant == param.type) {
            assert(gParamType_PrimVariant == param.defl.type().hash_code());
            const zeno::PrimVar& editVar = zeno::reflect::any_cast<zeno::PrimVar>(param.defl);
            bool bFind = std::visit([=](auto&& arg)->bool {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    if (arg.find("$F") != std::string::npos) {
                        return true;
                    }
                }
                return false;
            }, editVar);
            if (bFind)
                return true;
        }
        else if (gParamType_Vec2f == param.type ||
            gParamType_Vec2i == param.type ||
            gParamType_Vec3f == param.type ||
            gParamType_Vec3i == param.type ||
            gParamType_Vec4f == param.type ||
            gParamType_Vec4i == param.type)
        {
            assert(gParamType_VecEdit == param.defl.type().hash_code());
            const zeno::vecvar& editVar = zeno::reflect::any_cast<zeno::vecvar>(param.defl);
            for (auto primvar : editVar) {
                bool bFind = std::visit([=](auto&& arg)->bool {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::string>) {
                        if (arg.find("$F") != std::string::npos) {
                            return true;
                        }
                    }
                    return false;
                }, primvar);
                if (bFind)
                    return true;
            }
        }
    }
    return false;
}

bool NodeImpl::isInDopnetwork()
{
 /*   Graph* parentGraph = m_pGraph;
    while (parentGraph)
    {
        if (auto pSubnetImpl = parentGraph->getParentSubnetNode())
        {
            if (SubnetNode* subnet = getSubnetNode(pSubnetImpl)) {
                if (DopNetwork* dop = dynamic_cast<DopNetwork*>(subnet)) {
                    return true;
                }
                else {
                    parentGraph = pSubnetImpl->getGraph();
                }
            }
            else {
                break;
            }
        }
        else break;
    }*/
    return false;
}

void NodeImpl::onInterrupted() {
    mark_dirty(true);
    mark_previous_ref_dirty();
}

void NodeImpl::mark_clean() {
    m_status = Node_Clean;
    reportStatus(false, m_status);
}

void NodeImpl::mark_dirty(bool bOn, DirtyReason reason, bool bWholeSubnet, bool bRecursively)
{
    scope_exit sp([&] {
        m_status = Node_DirtyReadyToRun;  //修改了数据，标脏，并置为此状态。（后续在计算过程中不允许修改数据，所以markDirty理论上是前端驱动）
        reportStatus(m_dirty, m_status);
    });

    //有部分下游节点因为某些原因没有标脏，而上游节点已经脏了的情况下不会继续传播，所以不检查缓存
    //还是要检查，否则子图比较大的时候传播脏位会比较慢
    if (m_dirty == bOn)
        return;

    m_dirty = bOn;

    if (!bRecursively)
        return;

    if (m_dirty) {
        m_takenover = false;
        for (auto& [name, param] : m_inputPrims) {
            for (auto link : param.reflinks) {
                if (link->dest_inparam != &param) {
                    assert(link->dest_inparam);
                    auto destNode = link->dest_inparam->m_wpNode;
                    destNode->mark_dirty(true);
                }
            }
        }
        for (auto& [name, param] : m_outputObjs) {
            for (auto link : param.reflinks) {
                assert(link->dest_inparam);
                auto destNode = link->dest_inparam->m_wpNode;
                destNode->mark_dirty(true);
            }
            for (auto link : param.links) {
                auto inParam = link->toparam;
                assert(inParam);
                if (inParam) {
                    auto inNode = inParam->m_wpNode;
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
                    auto inNode = inParam->m_wpNode;
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
        //if (DopNetwork* pDop = dynamic_cast<DopNetwork*>(pSubnetNode)) {
        //    pDop->resetFrameState();
        //}
    }

    Graph* spGraph = m_pGraph;
    assert(spGraph);
    if (auto pSubnetImpl = spGraph->getParentSubnetNode())
    {
        pSubnetImpl->mark_dirty(true, Dirty_All, false);
    }

    dirty_changed(bOn, reason, bWholeSubnet, bRecursively);
}

void NodeImpl::dirty_changed(bool bOn, DirtyReason reason, bool bWholeSubnet, bool bRecursively) {

}

void NodeImpl::mark_dirty_objs()
{
    for (auto const& [name, param] : m_outputObjs)
    {
        if (param.spObject) {
            assert(param.spObject);
            if (param.spObject->key().empty()) {
                continue;
            }
        }
    }
}

void NodeImpl::complete() {}

void NodeImpl::preApply(CalcContext* pContext) {
    if (!m_dirty)
        return;

    //debug
#if 1
    if (m_name == "FormSceneTree1") {
        int j;
        j = 0;
    }
#endif

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

    //wait all
    for (auto& [name, param] : m_inputObjs) {
        for (auto link : param.links) {
            auto& task = link->upstream_task;
            if (task.valid()) {
                task.wait();
            }
        }
    }
    for (auto& [name, param] : m_inputPrims) {
        for (auto link : param.links) {
            auto& task = link->upstream_task;
            if (task.valid()) {
                task.wait();
            }
        }
    }

    //resolve all dependencys for input params
    for (auto& [name, param] : m_inputObjs) {
        if (param.type == gParamType_List) {
            param.spObject = processList(&param, pContext);
        }
        else if (param.type == gParamType_Dict) {
            param.spObject = processDict(&param, pContext);
        }
        else {
            if (param.links.size() == 1) {
                auto spLink = *param.links.begin();
                auto& task = spLink->upstream_task;
                if (task.valid()) {
                    param.spObject = task.get();
                }
                else {
                    if (!spLink->fromparam->spObject)
                        throw makeNodeError<UnimplError>(get_path(), "cannot get object from upstream");
                    //task为空，因为之前已经被执行过一次了，上游已经算好了
                    param.spObject = spLink->fromparam->spObject->clone();
                }
                param.spObject->update_key(stdString2zs(m_uuid));
            }
        }
    }

    for (auto& [name, param] : m_inputPrims) {
        if (param.type != gParamType_ListOfMat4) {
            //如果不是list类型，只会有一个连接
            if (!param.links.empty()) {
                auto spLink = *param.links.begin();
                auto& task = spLink->upstream_task;
                if (task.valid()) {
                    param.result = task.get();
                }
                else {
                    //直接从上游拷过来即可
                    param.result = spLink->fromparam->result;
                }
            }
        }
        else {
            std::vector<glm::mat4> result;
            for (auto spLink : param.links) {
                auto& task = spLink->upstream_task;
                zeno::reflect::Any res;
                if (task.valid()) {
                    res = task.get();
                }
                else {
                    res = spLink->fromparam->result;
                }
                if (res.type().hash_code() == gParamType_ListOfMat4) {
                    result = zeno::reflect::any_cast<std::vector<glm::mat4>>(res);
                    break;
                }
                else if (res.type().hash_code() == gParamType_Matrix4) {
                    result.push_back(zeno::reflect::any_cast<glm::mat4>(res));
                }
                else {
                    throw makeNodeError<UnimplError>(get_path(), "");
                }
            }
            param.result = result;
        }
    }
}

void NodeImpl::launch_param_task(const std::string& param) {
    auto iter = m_inputObjs.find(param);
    if (iter != m_inputObjs.end()) {
        ObjectParam& objParam = iter->second;
        if (!objParam.links.empty()) {
            auto spLink = *objParam.links.begin();
            auto& task = spLink->upstream_task;
            if (task.valid()) {
                //如果没有边，那么spObject在requireInput就会被清空
                objParam.spObject = task.get();
            }
            else {
                objParam.spObject = spLink->fromparam->spObject->clone();
            }
        }
    }
    else {
        auto iter2 = m_inputPrims.find(param);
        if (iter2 != m_inputPrims.end()) {
            PrimitiveParam& primParam = iter2->second;
            if (primParam.links.size() == 1) {
                auto spLink = *primParam.links.begin();
                auto& task = spLink->upstream_task;
                if (task.valid()) {
                    primParam.result = task.get();
                }
            }
        }
    }
}

void NodeImpl::preApply_SwitchIf(CalcContext* pContext) {
    preApply_Primitives(pContext);

    const zeno::reflect::Any& res = this->get_param_result("Condition");
    int cond = 0;
    if (res.type().hash_code() == gParamType_Int) {
        cond = zeno::reflect::any_cast<int>(res);
    }
    else if (res.type().hash_code() == gParamType_Float) {
        cond = zeno::reflect::any_cast<float>(res);
    }
    else if (res.type().hash_code() == gParamType_Bool) {
        cond = zeno::reflect::any_cast<bool>(res);
    }

    const std::string& exec_param = (cond != 0) ? "If True" : "If False";
    requireInput(exec_param, pContext);
    launch_param_task(exec_param);
}

void NodeImpl::preApply_SwitchBetween(CalcContext* pContext) {
    preApply_Primitives(pContext);

    int cond = zeno::reflect::any_cast<int>(get_param_result("cond1"));
    if (cond != 0) {
        requireInput("b1", pContext);
        launch_param_task("b1");
        return;
    }

    cond = zeno::reflect::any_cast<int>(get_param_result("cond2"));
    if (cond != 0) {
        requireInput("b2", pContext);
        launch_param_task("b2");
        return;
    }

    cond = zeno::reflect::any_cast<int>(get_param_result("cond3"));
    if (cond != 0) {
        requireInput("b3", pContext);
        launch_param_task("b3");
        return;
    }

    cond = zeno::reflect::any_cast<int>(get_param_result("cond4"));
    if (cond != 0) {
        requireInput("b4", pContext);
        launch_param_task("b4");
        return;
    }

    cond = zeno::reflect::any_cast<int>(get_param_result("cond5"));
    if (cond != 0) {
        requireInput("b5", pContext);
        launch_param_task("b5");
        return;
    }

    cond = zeno::reflect::any_cast<int>(get_param_result("cond6"));
    if (cond != 0) {
        requireInput("b6", pContext);
        launch_param_task("b6");
        return;
    }

    cond = zeno::reflect::any_cast<int>(get_param_result("cond7"));
    if (cond != 0) {
        requireInput("b7", pContext);
        launch_param_task("b7");
        return;
    }

    cond = zeno::reflect::any_cast<int>(get_param_result("cond8"));
    if (cond != 0) {
        requireInput("b8", pContext);
        launch_param_task("b8");
        return;
    }
}

void NodeImpl::preApply_Primitives(CalcContext* pContext) {
    if (!m_dirty)
        return;
    for (const auto& [name, param] : m_inputPrims) {
        bool ret = requireInput(name, pContext);
        if (!ret)
            zeno::log_warn("the param {} may not be initialized", name);
    }

    for (auto& [name, param] : m_inputPrims) {
        if (param.type != gParamType_ListOfMat4) {
            //如果不是list类型，只会有一个连接
            if (!param.links.empty()) {
                auto& task = (*param.links.begin())->upstream_task;
                if (task.valid()) {
                    param.result = task.get();
                }
            }
        }
        else {
            //TODO:
        }
    }
}

void NodeImpl::preApplyTimeshift(CalcContext* pContext)
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
    propagateDirty(this, "$F");

    preApply(pContext);
}

void NodeImpl::foreachend_apply(CalcContext* pContext)
{
    //当前节点是ForeachEnd
    std::string foreach_begin_path = zeno::any_cast_to_string(get_defl_value("ForEachBegin Path"));
    if (Graph* spGraph = m_pGraph) {
        auto foreach_begin = spGraph->getNode(foreach_begin_path);
        auto foreach_end = static_cast<ForEachEnd*>(coreNode());
        assert(foreach_end);
        for (foreach_end->reset_forloop_settings(); foreach_end->is_continue_to_run(pContext); foreach_end->increment())
        {
            foreach_begin->mark_dirty(true);
            //pContext->curr_iter = zeno::reflect::any_cast<int>(foreach_begin->get_defl_value("Current Iteration"));

            preApply(pContext);
            foreach_end->apply_foreach(pContext);
        }
        auto output = get_output_obj("Output Object");
        if (output)
            output->update_key(stdString2zs(m_uuid));
    }
}


void NodeImpl::apply() {
    if (m_pNode) {
        try {
            m_pNode->apply();
        }
        catch (ErrorException const& e) {
            if (e.get_node_info().empty()) {
                throw ErrorException(get_path(), e.getError());
            }
            else {
                throw e;
            }
        }
        catch (std::exception const& e) {
            std::string err = e.what();
            throw makeNodeError<StdError>(get_path(), std::current_exception());
        }
        catch (...) {
            throw makeNodeError<UnimplError>(get_path(), "unknown error");
        }
    }
    else {
        throw makeNodeError<UnimplError>(get_path(), "the node has been uninstalled");
    }
}

void NodeImpl::reflectNode_apply()
{
    assert(false);
#if 0
    if (m_pTypebase) {
        for (zeno::reflect::IMemberFunction* func : m_pTypebase->get_member_functions()) {
            const auto& funcname = func->get_name();
            if (funcname == "apply") {
                //根据ReflectCustomUI获取fieldName到displayName映射
                std::map<std::string, std::string> mappingInputParams;
                std::vector<std::string> mappingReturnParams;

                zeno::_ObjectParam retInfoOnReflectUI;
                getNameMappingFromReflectUI(m_pTypebase, this, mappingInputParams, mappingReturnParams);

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
                    //类型不一致，数值类型可能需要转换
                    ParamType outtype = inputAny.type().hash_code();
                    ParamType intype = param_type.hash_code();
                    if (intype != outtype && isNumericVecType(outtype)) {
                        inputAny = convertNumericAnyType(outtype, intype, inputAny);
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
                                //类型不一致，数值类型可能需要转换
                                ParamType outtype = inputAny.type().hash_code();
                                ParamType intype = field->get_field_type().type_hash();
                                if (intype != outtype && isNumericVecType(outtype)) {
                                    inputAny = convertNumericAnyType(outtype, intype, inputAny);
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
#endif
}

void NodeImpl::update_out_objs_key()
{
    for (auto const& [name, param] : m_outputObjs)
    {
        if (param.spObject)
        {
            //目前节点处所看到的object，都隶属于此节点本身。
            if (param.spObject->key().empty())
                param.spObject->update_key(stdString2zs(m_uuid));
        }
    }
}

void NodeImpl::on_link_added_removed(bool bInput, const std::string& paramname, bool bAdded) {
    checkParamsConstrain();

    if (!bInput && bAdded) {
        auto iter = m_outputObjs.find(paramname);
    }
}

void NodeImpl::on_node_about_to_remove() {
    //移除所有引用边的依赖关系
    for (auto& [_, input_param] : m_inputPrims)
    {
        for (const std::shared_ptr<ReferLink>& reflink : input_param.reflinks) {
            if (reflink->source_inparam == &input_param) {
                //当前参数是引用源
                auto& otherLinks = reflink->dest_inparam->reflinks;
                otherLinks.erase(std::remove(otherLinks.begin(), otherLinks.end(), reflink));
                //参数值也改掉吧，把ref(...)改为 inv_ref(...)
                auto otherNode = reflink->dest_inparam->m_wpNode;
                assert(otherNode);

                auto defl = reflink->dest_inparam->defl;
                assert(defl.has_value());
                ParamType type = defl.type().hash_code();
                if (type == gParamType_PrimVariant) {
                    PrimVar var = any_cast<PrimVar>(defl);
                    std::visit([&](auto& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, std::string>) {
                            auto iter = arg.find("ref(");
                            if (iter != std::string::npos) {
                                arg.replace(arg.find("ref("), 4, "ref_not_exist(");
                            }
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
                                auto iter = arg.find("ref(");
                                if (iter != std::string::npos) {
                                    arg.replace(iter, 4, "ref_not_exist(");
                                }
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
                auto otherNode = reflink->source_inparam->m_wpNode;
                if (otherNode)
                    otherNode->mark_dirty(true);
            }
        }
        input_param.reflinks.clear();
    }
    for (auto& [_, output_obj] : m_outputObjs)
    {
        for (const std::shared_ptr<ReferLink>& reflink : output_obj.reflinks) {
            if (reflink->source_inparam == &output_obj) {
                //当前参数是引用源
                auto& otherLinks = reflink->dest_inparam->reflinks;
                otherLinks.erase(std::remove(otherLinks.begin(), otherLinks.end(), reflink));

                auto otherNode = reflink->dest_inparam->m_wpNode;
                assert(otherNode);
                if (otherNode)
                    otherNode->mark_dirty(true);
            }
        }
    }
}

void NodeImpl::onNodeNameUpdated(const std::string& oldname, const std::string& newname) {
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
                    PrimVar var = zeno::reflect::any_cast<PrimVar>(adjustParamVal);
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
                    auto spDestNode = reflink->dest_inparam->m_wpNode;
                    spDestNode->update_param(reflink->dest_inparam->name, adjustParamVal);
                }
            }
        }
    }
}

void NodeImpl::constructReference(const std::string& param_name) {
    auto iter = m_inputPrims.find(param_name);
    if (iter == m_inputPrims.end())
        return;

    const Any& param_defl = iter->second.defl;
    initReferLinks(&iter->second);
}

void NodeImpl::initReferLinks(PrimitiveParam* target_param) {
    if (m_pGraph->isAssetRoot()) {
        //资产图不会执行，无须构造引用关系
        return;
    }
    std::set<std::pair<std::string, std::string>> refSources = resolveReferSource(target_param->defl);
    auto newAdded = refSources;

    for (auto iter = target_param->reflinks.begin(); iter != target_param->reflinks.end(); )
    {
        bool bExist = false;
        std::shared_ptr<ReferLink> spRefLink = (*iter);
        CoreParam* remote_source = spRefLink->source_inparam;
        assert(remote_source);
        if (remote_source == target_param) {
            iter++;
            continue;
        }

        //查看当前link在新的集合里是否还存在。
        for (const auto& [source_node_uuidpath, source_param] : refSources)
        {
            auto spSrcNode = remote_source->m_wpNode;
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
        //目前引用功能只支持本图引用，不能跨图引用
        std::string sourcenode_uuid;
        if (source_node_uuidpath.find('/') != std::string::npos) {
            sourcenode_uuid = source_node_uuidpath.substr(source_node_uuidpath.find_last_of('/') + 1);
        }
        else {
            sourcenode_uuid = source_node_uuidpath;
        }

        NodeImpl* srcNode = getGraph()->getNodeByUuidPath(sourcenode_uuid);
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
        } else {
            auto iterSrcObj = srcNode->m_outputObjs.find(source_param);
            if (iterSrcObj != srcNode->m_outputObjs.end()) {
                ObjectParam& srcObj = iterSrcObj->second;
                //构造reflink
                std::shared_ptr<ReferLink> reflink = std::make_shared<ReferLink>();
                reflink->source_inparam = &srcObj;
                reflink->dest_inparam = target_param;
                target_param->reflinks.push_back(reflink);
                srcObj.reflinks.push_back(reflink);
            }
            else {
                auto iterOutPrim = srcNode->m_outputPrims.find(source_param);
                if (iterOutPrim != srcNode->m_outputPrims.end()) {
                    PrimitiveParam& srcparam = iterOutPrim->second;
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
}

std::set<std::pair<std::string, std::string>> NodeImpl::resolveReferSource(const Any& param_defl) {

    std::set<std::pair<std::string, std::string>> refSources;
    std::vector<std::string> refSegments;

    ParamType deflType = param_defl.type().hash_code();
    if (deflType == zeno::types::gParamType_String) {
        const std::string& param_text = zeno::any_cast_to_string(param_defl);
        if (param_text.find("ref") != std::string::npos) {
            refSegments.push_back(param_text);
        }
    }
    else if (deflType == zeno::types::gParamType_PrimVariant) {
        zeno::PrimVar var = zeno::reflect::any_cast<zeno::PrimVar>(param_defl);
        if (!std::holds_alternative<std::string>(var)) {
            return refSources;
        }
        std::string param_text = std::get<std::string>(var);
        if (param_text.find("ref") != std::string::npos) {
            refSegments.push_back(param_text);
        }
    }
    else if (deflType == zeno::types::gParamType_VecEdit) {
        const zeno::vecvar& vec = zeno::reflect::any_cast<zeno::vecvar>(param_defl);
        for (const zeno::PrimVar& elem : vec) {
            if (!std::holds_alternative<std::string>(elem)) {
                continue;
            }
            std::string param_text = std::get<std::string>(elem);
            if (param_text.find("ref") != std::string::npos) {
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
        FunctionManager funcMgr;
        ZfxContext ctx;
        ctx.spNode = this;
        for (auto param_text : refSegments)
        {
            std::string code = param_text;
            if (param_text.find('\n') == std::string::npos && param_text.back() != ';') {
                code = param_text + ';';    //方便通过zfx语句编译
            }

            ZfxContext ctx;
            ctx.spNode = this;
            ctx.spObject = nullptr;
            ctx.code = code;
            ZfxExecute zfx(code, &ctx);
            int ret = zfx.parse();
            if (ret == 0)
            {
                ctx.code = code;
                std::shared_ptr<ZfxASTNode> astRoot = zfx.getASTResult();
                std::set<std::pair<std::string, std::string>> paths =
                    funcMgr.getReferSources(astRoot, &ctx);
                if (!paths.empty()) {
                    refSources.insert(paths.begin(), paths.end());
                }
            }
        }
    }, err);
    return refSources;
}

std::unique_ptr<DictObject> NodeImpl::processDict(ObjectParam* in_param, CalcContext* pContext) {
    //连接的元素是list还是list of list的规则，参照Graph::addLink下注释。
    bool bDirecyLink = false;
    const auto& inLinks = in_param->links;
#if 0
    if (inLinks.size() == 1)
    {
        std::shared_ptr<ObjectLink> spLink = inLinks.front();
        auto out_param = spLink->fromparam;
        std::shared_ptr<INode> outNode = out_param->m_wpNode;
        assert(outNode);

        if (out_param->type == in_param->type && spLink->tokey.empty()) //根据Graph::addLink规则，类型相同且无key视为直连
        {
            bDirecyLink = true;
            if (outNode->is_dirty()) {
                GraphException::translated([&] {
                    outNode->doApply(pContext);
                    }, outNode.get());
            }
            auto outResult = outNode->get_output_obj(out_param->name);
            assert(outResult);
            assert(out_param->type == gParamType_Dict);

            //规则如普通节点，都是直接拷贝
#if 0
            if (in_param->socketType == Socket_Owning) {
                spDict = std::dynamic_pointer_cast<DictObject>(outResult->move_clone());
            }
            else if (in_param->socketType == Socket_ReadOnly) {
                spDict = std::dynamic_pointer_cast<DictObject>(outResult);
            }
            else if (in_param->socketType == Socket_Clone)
#endif
            {
                //里面的元素也要clone
                std::shared_ptr<DictObject> outDict = std::dynamic_pointer_cast<DictObject>(outResult);
                spDict = std::dynamic_pointer_cast<DictObject>(outDict->clone_by_key(m_uuid));
            }
            spDict->update_key(m_uuid);
            return spDict;
        }
    }
#endif
    if (!bDirecyLink)
    {
        std::map<std::string, IObject*> existObjs;
        if (in_param->spObject) {
            auto spOldDict = dynamic_cast<DictObject*>(in_param->spObject.get());
            for (auto& [key, spobj] : spOldDict->lut) {
                std::string skey = zsString2Std(spobj->key());
                existObjs.insert({skey, spobj.get()});
            }
        }

        auto spDict = std::make_unique<DictObject>();
        for (const auto& spLink : in_param->links)
        {
            const std::string& keyName = spLink->tokey;
            auto out_param = spLink->fromparam;
            zeno::NodeImpl* outNode = out_param->m_wpNode;

            bool is_this_item_dirty = outNode->is_dirty();

            if (is_this_item_dirty) {  //list中的元素是dirty的，重新计算并加入list
                outNode->doApply(pContext);
            }

            auto outResult = outNode->get_output_obj(out_param->name);
            assert(outResult);
            {
                zany newObj;
                if (auto _spList = dynamic_cast<zeno::ListObject*>(outResult)) {
                    newObj = clone_by_key(_spList, m_uuid);
                }
                else if (auto _spDict = dynamic_cast<zeno::DictObject*>(outResult)) {
                    newObj = clone_by_key(_spDict, m_uuid);
                }
                else {
                    zeno::String newkey = stdString2zs(m_uuid) + '\\' + outResult->key();
                    newObj = outResult->clone();
                    newObj->update_key(newkey);
                }
                
                std::string const& new_key = zsString2Std(newObj->key());
                spDict->lut[keyName] = std::move(newObj);

                if (is_this_item_dirty) {
                    //需要区分是新的还是旧的，这里先粗暴认为全是新的
                    if (existObjs.find(new_key) != existObjs.end()) {
                        spDict->m_modify.insert(new_key);
                        existObjs.erase(new_key);
                    }
                    else {
                        spDict->m_new_added.insert(new_key);
                    }
                } else {
                    spDict->m_modify.insert(new_key);//需视为modify，否则关闭这个list节点的view时会崩或显示不正常的bug
                }
                existObjs.erase(new_key);
            }
        }
        //剩下没出现的都认为是移除掉了
        std::function<void(IObject*)> flattenDict = [&flattenDict, &spDict](IObject* obj) {
            if (auto _spList = dynamic_cast<ListObject*>(obj)) {
                for (int i = 0; i < _spList->m_impl->size(); ++i) {
                    flattenDict(_spList->m_impl->get(i));
                }
            } else if (auto _spDict = dynamic_cast<DictObject*>(obj)) {
                for (auto& [key, obj] : _spDict->get()) {
                    flattenDict(obj);
                }
            } else {
                spDict->m_new_removed.insert(zsString2Std(obj->key()));
            }
        };
        for (auto& [key, removeobj] : existObjs) {
            flattenDict(removeobj);//提前节需要移除的对象的key全部展平，在GraphicsManager.h中一次性移除，才能正确显示
        }

        spDict->update_key(stdString2zs(m_uuid));
        return spDict;
    }
    return nullptr;
}

void NodeImpl::clear_container_info() {
    for (auto& [_, param] : m_inputObjs) {
        for (auto spLink : param.links) {
            spLink->bTraceAndTaken = false;
        }
        if (param.type == gParamType_List && param.spObject) {
            auto spList = static_cast<ListObject*>(param.spObject.get());
            spList->m_impl->m_new_added.clear();
            spList->m_impl->m_new_removed.clear();
            spList->m_impl->m_modify.clear();
        }
    }
    for (auto& [_, param] : m_outputObjs) {
        for (auto spLink : param.links) {
            spLink->bTraceAndTaken = false;
        }
        if (param.type == gParamType_List && param.spObject) {
            auto spList = static_cast<ListObject*>(param.spObject.get());
            spList->m_impl->m_new_added.clear();
            spList->m_impl->m_new_removed.clear();
            spList->m_impl->m_modify.clear();
        }
    }
}

std::unique_ptr<ListObject> NodeImpl::processList(ObjectParam* in_param, CalcContext* pContext) {
    assert(gParamType_List == in_param->type);
    
    bool bDirectLink = false;

    if (m_nodecls == "FormSceneTree") {
        int j;
        j = 0;
    }

    if (in_param->links.size() == 1)
    {
        std::shared_ptr<ObjectLink> spLink = in_param->links.front();
        auto out_param = spLink->fromparam;
        NodeImpl* outNode = out_param->m_wpNode;

        auto list_register_all_items = [&](ListObject* listobj) {
            if (!listobj->has_change_info()) {
                //上游没有修改信息，只能全部加进来，还要比较有哪些被删掉
                for (const auto& obj : listobj->m_impl->m_objects) {
                    std::string key = zsString2Std(obj->key());
                    if (key.empty()) throw makeNodeError<UnimplError>(get_path(), "there is object in list with empty key");
                    //直接全部收集
                    listobj->m_impl->m_new_added.insert(key);
                }
            }
        };

        if (outNode->is_takenover()) {
            //可能上一次计算被taken了
            return nullptr;
        }

        if ((out_param->type == in_param->type || out_param->type == gParamType_IObject) &&
            spLink->tokey.empty())
        {
            std::unique_ptr<ListObject> spList;
            bool bAllTaken = false;     //输出参数所有的链路（包括本链路）都被获取了
            if (spLink->upstream_task.valid()) {
                auto outResult = spLink->upstream_task.get();   //outResult已经是本节点输入参数所有，不属于outnode了
                auto _spList = dynamic_cast<ListObject*>(outResult.get());
                if (!_spList) {
                    throw makeNodeError(get_path(), "no list object received");
                }
                spList.reset(static_cast<ListObject*>(outResult.release()));
                //无论原来有没有缓存，上游的list已经脏了，就干脆直接换新的，不去一个个比较了
                list_register_all_items(spList.get());
                spList->update_key(out_param->spObject->key());
            }
            else {
                assert(!outNode->is_dirty());
                //上游已经算好了，但当前的输入没有建立缓存，就得从上游拷贝一下
                if (in_param->spObject) {
                    //相当于转一次又给回外面。。。
                    spList = safe_uniqueptr_cast<ListObject>(std::move(in_param->spObject));
                }
                else {
                    //outNode已经算好了，直接拿应该不会导致race condition
                    spList = safe_uniqueptr_cast<ListObject>(out_param->spObject->clone());
                    //新的list，这里全部内容都要登记到new_added.
                    list_register_all_items(spList.get());
                    spList->update_key(out_param->spObject->key());
                }
            }
            if (!spList) {
                throw makeNodeError<UnimplError>(get_path(), "no outResult List from output");
            }
            bDirectLink = true;
            add_prefix_key(spList.get(), m_uuid);
            return spList;
        }
    }
    if (!bDirectLink)
    {
        auto spList = create_ListObject();
        auto cachedList = static_cast<ListObject*>(in_param->spObject.get());
        std::set<std::string> old_list;
        if (cachedList) {
            for (const auto& obj : cachedList->m_impl->m_objects) {
                old_list.insert(zsString2Std(obj->key()));
            }
        }

        //TODO: 多个物体作为List的元素，如果其中有一个被take over，要怎么处理？
        for (const auto& spLink : in_param->links)
        {
            auto out_param = spLink->fromparam;
            std::string upstream_obj_key;
            std::string new_obj_key;
            NodeImpl* outNode = out_param->m_wpNode;
            if (outNode->is_takenover()) {
                continue;
            }
            if (spLink->upstream_task.valid()) {
                //有任务发起，说明上游新增了或者修改了某一个节点
                auto outResult = spLink->upstream_task.get();
                upstream_obj_key = zsString2Std(out_param->spObject->key());
                outResult->update_key(stdString2zs(upstream_obj_key));
                add_prefix_key(outResult.get(), m_uuid);
                new_obj_key = zsString2Std(outResult->key());

                spList->push_back(std::move(outResult));

                if (old_list.find(new_obj_key) != old_list.end()) {
                    //旧的列表有这一项，说明是修改的
                    spList->m_impl->m_modify.insert(new_obj_key);
                }
                else {
                    //旧的列表没有，现在发现新的
                    spList->m_impl->m_new_added.insert(new_obj_key);
                }
            }
            else {
                //没有任务发起，但上游可能已经有缓存好的结果，直接加到list即可
                assert(!outNode->is_dirty());
                //想知道是不是新增，要与oldList对比
                upstream_obj_key = zsString2Std(out_param->spObject->key());
                new_obj_key = m_uuid + '\\' + upstream_obj_key;
                if (old_list.find(new_obj_key) != old_list.end()) {
                    //直接从缓存取就行
                    for (const auto& obj : cachedList->m_impl->m_objects) {
                        if (zsString2Std(obj->key()) == new_obj_key) {
                            spList->push_back(obj->clone());
                            break;
                        }
                    }
                }
                else {
                    //上游节点不脏，但边新增到list，这时候需要标脏这个obj，否则渲染端没法认出
                    auto new_obj = out_param->spObject->clone();
                    new_obj->update_key(stdString2zs(upstream_obj_key));
                    add_prefix_key(new_obj.get(), m_uuid);
                    spList->m_impl->m_modify.insert(zsString2Std(new_obj->key()));

                    spList->push_back(std::move(new_obj));
                }
            }
            old_list.erase(new_obj_key);
        }
        //从old_list剩下的，就是要被删除的元素
        spList->m_impl->m_new_removed = old_list;
        spList->update_key(stdString2zs(m_uuid));
        return spList;
    }
    return nullptr;
}

zeno::reflect::Any NodeImpl::processPrimitive(PrimitiveParam* in_param)
{
    //没有连线的情况，这时候直接取defl作为result。
    if (!in_param) {
        return Any();
    }

    int frame = getGlobalState()->getFrameId();

    const std::string& name = in_param->name;
    const ParamType type = in_param->type;
    const auto& defl = in_param->defl;
    if (type == gParamType_Heatmap && !defl.has_value()) {
        //先跳过heatmap
        return Any();
    }
    assert(defl.has_value());
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
                    throw makeNodeError<UnimplError>(get_path(), "error type with `gParamType_PrimVariant`");
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
        if (in_param->control != Lineedit &&
            in_param->control != Multiline &&
            in_param->control != ReadPathEdit &&
            in_param->control != WritePathEdit &&
            in_param->control != NullControl/*初始化的时候只有type没有control，而ui会自动初始化*/) {
            //不可能是公式
            break;
        }
        if (in_param->control == CodeEditor) {
            //单独在wrangle里parse execute.
            break;
        }

        //有很多ref子图中字符串类型的参数，所以一切字符串都要parse
        const std::string& code = any_cast<std::string>(defl);
        if (!code.empty()) {
            result = resolve_string(code, code);
        }
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
                    throw makeNodeError<UnimplError>(get_path());
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
    //case zeno::types::gParamType_Heatmap:
    //{
    //    //TODO: heatmap的结构体定义.
    //    //if (std::holds_alternative<std::string>(defl))
    //    //    result = zeno::parseHeatmapObj(std::get<std::string>(defl));
    //    break;
    //}
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
    case gParamType_ListOfMat4:
    {
        if (in_param->links.size() == 1 && in_param->links.front()->toparam->type == gParamType_ListOfMat4) {
            //TODO
        }
        else {
            for (auto link : in_param->links) {
                
            }
        }
        break;
    }
    case gParamType_Shader:
    {
        throw makeNodeError<UnimplError>(get_path(), "no defl value supported on the param with type `gParamType_Shader`, please connect the link by outside");
#if 0
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
                    throw makeNodeError<UnimplError>(get_path());
                }
            }, var);
        }
        else if (editType == gParamType_Int) {
            result = defl;
        }
        else if (editType == gParamType_Float) {
            result = defl;
        }
        break;
#endif
    }
    case gParamType_Curve:
    {
        result = defl;
        break;
    }
    }
    return result;
}

bool NodeImpl::receiveOutputObj(ObjectParam* in_param, NodeImpl* outNode, ObjectParam* out_param) {

    //在此版本里，只有克隆，每个对象只有一个节点关联，虽然激进，但可以充分测试属性数据共享在面对
    //内存暴涨时的冲击，能优化到什么程度

    bool bCloned = true;
    //如果outputobj是 PrimitiveObject，则不走拷贝，而是move，如果List和Dict含有哪怕只有一个PrimitiveObj，也不走拷贝
    if (auto prim = dynamic_cast<PrimitiveObject*>(out_param->spObject.get())) {
        bCloned = false;
    }
    else if (auto lst = dynamic_cast<ListObject*>(out_param->spObject.get())) {
        if (ListHasPrimObj(lst)) {
            bCloned = false;
        }
    }
    else if (auto dict = dynamic_cast<DictObject*>(out_param->spObject.get())) {
        if (DictHasPrimObj(dict)) {
            bCloned = false;
        }
    }

    bool bAllTaken = false;
    auto outputObj = outNode->takeOutputObject(out_param, in_param, bAllTaken);
    in_param->spObject = outputObj->clone();
    in_param->spObject->update_key(stdString2zs(m_uuidPath));

    if (outNode->is_nocache() && bAllTaken) {
        //似乎要把输入也干掉，但如果一锅清掉，可能会漏了数值输出，或者多对象输出的情况（虽然很少见）
        outNode->mark_takeover();
    }

    if (auto splist = dynamic_cast<ListObject*>(in_param->spObject.get())) {
        update_list_root_key(splist, m_uuidPath);
    } else if (auto spdict = dynamic_cast<DictObject*>(in_param->spObject.get())) {
        update_dict_root_key(spdict, m_uuidPath);
    }

    return true;
}

void NodeImpl::execute(CalcContext* pContext) {
    std::lock_guard scope(m_mutex);
    doApply(pContext);
}

zany NodeImpl::execute_get_object(const ExecuteContext& exec_context) {
    //锁的粒度可以更精细化，比如没有apply和takeover，在只读的情况下，可以释放锁
    //另一方面，如果多个节点想获取同一个节点的输出，就得排队了
    std::lock_guard scope(m_mutex);

    doApply(exec_context.pContext);

    //这里先不考虑takeout的情况，因为这使得临界区变得复杂
    bool bAllTaken = false;
    auto result = get_output_obj(exec_context.out_param);
    //zany result = takeOutputObject(exec_context.out_param, exec_context.in_param, bAllTaken);
    if (!result) {
        throw makeNodeError<UnimplError>(get_path(), "no result");
    }

    if (bAllTaken && is_nocache()) {
        //TODO: 是否要在这里清理outputNode?
        //其实可以，因为这里并不是存粹的数学函数，必然有修改当前节点状态的可能。
        mark_takeover();
    }
    auto spRes = result->clone();
    return spRes;
}

zeno::reflect::Any NodeImpl::execute_get_numeric(const ExecuteContext& exec_context) {
    std::lock_guard scope(m_mutex);

    doApply(exec_context.pContext);

    bool bAllOutputTaken = false;
    auto result = takeOutputPrim(exec_context.out_param, exec_context.in_param, bAllOutputTaken);
    if (bAllOutputTaken && is_nocache()) {
        mark_takeover();
    }
    return result;
}

bool NodeImpl::requireInput(std::string const& ds, CalcContext* pContext) {
    // 目前假设输入对象和输入数值，不能重名（不难实现，老节点直接改）。
    auto launch_method = zeno::getSession().is_async_executing() ? 
        (std::launch::async | std::launch::deferred) : std::launch::deferred;

    if (ds == "path") {
        int j;
        j = 0;
    }

    auto iter = m_inputObjs.find(ds);
    if (iter != m_inputObjs.end()) {
        ObjectParam* in_param = &(iter->second);
        if (in_param->links.empty()) {
            //节点如果定义了对象，但没有边连上去，是否要看节点apply如何处理？
            //FIX: 没有边的情况要清空掉对象，否则apply以为这个参数连上了对象
            in_param->spObject.reset();
        }
        else {
            if (in_param->links.empty()) {
                //清空缓存对象
                in_param->spObject.reset();
            }

            //改为异步计算以后，直接发起task即可，后续再考虑组装的事宜
            for (auto spLink : in_param->links) {
                ObjectParam* out_param = spLink->fromparam;
                auto outNode = out_param->m_wpNode;
                if (outNode->is_takenover()) {
                    //维持原来的参数结果状态，不从上游取（事实上上游的内容已经被删掉了）
                    return true;
                }
                if (outNode->is_dirty())
                {
                    ExecuteContext ctx;
                    ctx.in_node = m_name;
                    ctx.in_param = in_param->name;
                    ctx.out_param = out_param->name;
                    ctx.pContext = pContext;
                    ctx.innode_uuid_path = m_uuidPath;

                    //发起异步任务：
                    spLink->upstream_task = std::async(launch_method, &NodeImpl::execute_get_object, outNode, ctx);
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
                        auto spSrcNode = reflink->source_inparam->m_wpNode;
                        assert(spSrcNode);
                        Graph* spSrcGraph = spSrcNode->m_pGraph;
                        assert(spSrcGraph);
                        //NOTE in 2025/3/25: 还是apply引用源，至于本节点参数循环引用的问题，走另外的路线
                        //TODO: refactor with async tasks
                        if (spSrcNode == this) {
                            //引用自身的参数，直接拿defl，因为这种情况绝大多数是固定值，没必要执行计算，比如pos引用size的数据
                            spSrcNode->doApply_Parameter(reflink->source_inparam->name, pContext);
                        }
                        else {
                            spSrcNode->doApply(pContext);
                        }
                    }
                }

                const zeno::reflect::Any& primval = processPrimitive(in_param);
                in_param->result = primval;
                //旧版本的requireInput指的是是否有连线，如果想兼容旧版本，这里可以返回false，但使用量不多，所以就修改它的定义。
            }
            else {
                for (auto spLink : in_param->links) {
                    auto outNode = spLink->fromparam->m_wpNode;
                    if (outNode->is_takenover()) {
                        continue;
                    }
                    if (outNode->is_dirty())
                    {
                        ExecuteContext ctx;
                        ctx.in_node = m_name;
                        ctx.in_param = in_param->name;
                        ctx.out_param = spLink->fromparam->name;
                        ctx.pContext = pContext;
                        ctx.innode_uuid_path = m_uuidPath;
                        spLink->upstream_task = std::async(launch_method, &NodeImpl::execute_get_numeric, outNode, ctx);
                    }
                    else {
                        //直接从上游拷过来即可
                        in_param->result = spLink->fromparam->result;
                    }
                }
                
            }
        } else {
            return false;
        }
    }
    return true;
}

void NodeImpl::doOnlyApply() {
    apply();
}

float NodeImpl::time() const {
    if (m_pNode) return m_pNode->time();
    return 0;
}

void NodeImpl::clearCalcResults() {
    if (m_pNode)
        m_pNode->clearCalcResults();

    for (auto& [key, param] : m_inputObjs) {
        param.spObject.reset();
    }
    for (auto& [key, param] : m_outputObjs) {
        param.spObject.reset();
    }
    for (auto& [key, param] : m_inputPrims) {
        param.result.reset();
    }
    for (auto& [key, param] : m_outputPrims) {
        param.result.reset();
    }
}

void NodeImpl::doApply_Parameter(std::string const& name, CalcContext* pContext) {
    if (!m_dirty) {
        return;
    }

    std::string uuid_path = get_uuid_path() + "/" + name;
    if (pContext->uuid_node_params.find(uuid_path) != pContext->uuid_node_params.end()) {
        throw makeNodeError<UnimplError>(get_path(), "cycle reference occurs when refer paramters!");
    }

    scope_exit scope_apply_param([&]() { pContext->uuid_node_params.erase(uuid_path); });
    pContext->uuid_node_params.insert(uuid_path);

    requireInput(name, pContext);
}

void NodeImpl::bypass() {
    //在preAppy拷贝了对象以后再直接赋值给output，还有一种方法是不拷贝，直接把上一个节点的输出给到这里的输出。

    //找到输入和输出的唯一object(如果输入有两个，并且有一个有连线，是否采纳连线这个？）
    //不考虑数值类型的输出
    if (m_outputObjs.empty() || m_inputObjs.empty()) {
        throw makeNodeError<UnimplError>(get_path(), "there is not matched input and output object when mute button is on");
    }
    ObjectParam& input_objparam = m_inputObjs.begin()->second;
    ObjectParam& output_objparam = m_outputObjs.begin()->second;
    if (input_objparam.type != output_objparam.type) {
        throw makeNodeError<UnimplError>(get_path(), "the input and output type is not matched, when the mute button is on");
    }
    output_objparam.spObject = input_objparam.spObject->clone();
}

void NodeImpl::check_break_and_return() {
    if (zeno::getSession().is_interrupted()) {
        m_status = Node_DirtyReadyToRun;
        reportStatus(m_dirty, m_status);
        throw makeNodeError<InterruputError>(this->get_path(), get_path());
    }
}

void NodeImpl::doApply(CalcContext* pContext) {
    if (!m_dirty)
        return;

    check_break_and_return();

    assert(pContext);
    std::string uuid_path = get_uuid_path();

    scope_exit spUuidRecord([=] {
        std::lock_guard scope(pContext->mtx);
        pContext->visited_nodes.erase(uuid_path);
        });

    {
        std::lock_guard scope(pContext->mtx);
        if (pContext->visited_nodes.find(uuid_path) != pContext->visited_nodes.end()) {
            throw makeNodeError<UnimplError>(get_path(), "cycle reference occurs!");
        }
        pContext->visited_nodes.insert(uuid_path);
    }

#if 1
    if (m_bypass) {
        set_name(m_name);
    }
    if (m_name == "MergeScene1") {//}&& pContext->curr_iter == 1) {
        set_name(m_name);
    }
#endif

    for (auto const& [name, param] : m_outputObjs) {
        if (param.type == gParamType_List && param.spObject) {
            auto list = static_cast<ListObject*>(param.spObject.get());
            list->m_impl->m_modify.clear();
            list->m_impl->m_new_added.clear();
            list->m_impl->m_new_removed.clear();
        }
    }

    if (m_nodecls == "TimeShift") {
        preApplyTimeshift(pContext);
    } else if (m_nodecls == "SwitchIf") {
        preApply_SwitchIf(pContext);
    } else if (m_nodecls == "SwitchBetween") {
        preApply_SwitchBetween(pContext);
    } else if (m_nodecls == "ForEachEnd") {
        preApply_Primitives(pContext);
    } else {
        preApply(pContext);
    }

    check_break_and_return();

    log_debug("==> enter {}", m_name);
    {
#ifdef ZENO_BENCHMARKING
        //Timer _(m_name);
#endif
        //暂时废弃bypass，先作为一个debug节点
        if (false && m_bypass) {
            bypass();
        }
        else {
            reportStatus(true, Node_Running);
            if (m_nodecls == "ForEachEnd") {
                foreachend_apply(pContext);
            }
            else {
                apply();
            }
        }
    }
    log_debug("==> leave {}", m_name);

    update_out_objs_key();
    reportStatus(false, Node_RunSucceed);
}

bool NodeImpl::is_upstream_dirty(const std::string& in_param) const {
    if (auto iter = m_inputObjs.find(in_param); iter != m_inputObjs.end()) {
        const auto& param = iter->second;
        for (auto spLink : param.links) {
            auto outNode = spLink->fromparam->m_wpNode;
            NodeRunStatus status = outNode->get_run_status();
            if (status != Node_Clean) {
                return true;
            }
        }
        return false;
    }
    else if (auto iter = m_inputPrims.find(in_param); iter != m_inputPrims.end()) {
        const auto& param = iter->second;
        for (auto spLink : param.links) {
            auto outNode = spLink->fromparam->m_wpNode;
            NodeRunStatus status = outNode->get_run_status();
            if (status != Node_Clean) {
                return true;
            }
        }
        return false;
    }
    else {
        throw makeNodeError<UnimplError>(get_path(), "the param is not exist");
    }
}

CommonParam NodeImpl::get_input_param(std::string const& name, bool* bExist) {
    auto primparam = get_input_prim_param(name, bExist);
    if (bExist && *bExist)
        return primparam;
    auto objparam = get_input_obj_param(name, bExist);
    if (bExist && *bExist)
        return objparam;
    if (bExist)
        *bExist = false;
    return CommonParam();
}

CommonParam NodeImpl::get_output_param(std::string const& name, bool* bExist) {
    auto primparam = get_output_prim_param(name, bExist);
    if (bExist && *bExist)
        return primparam;
    auto objparam = get_output_obj_param(name, bExist);
    if (bExist && *bExist)
        return objparam;
    if (bExist)
        *bExist = false;
    return CommonParam();
}

ObjectParams NodeImpl::get_input_object_params() const
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

ObjectParams NodeImpl::get_output_object_params() const
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

PrimitiveParams NodeImpl::get_input_primitive_params() const {
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

PrimitiveParams NodeImpl::get_output_primitive_params() const {
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

ParamPrimitive NodeImpl::get_input_prim_param(std::string const& name, bool* pExist) const {
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

ParamObject NodeImpl::get_input_obj_param(std::string const& name, bool* pExist) const {
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

ParamPrimitive NodeImpl::get_output_prim_param(std::string const& name, bool* pExist) const {
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

ParamObject NodeImpl::get_output_obj_param(std::string const& name, bool* pExist) const {
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

zeno::reflect::Any NodeImpl::get_defl_value(std::string const& name) {
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

zeno::reflect::Any NodeImpl::get_param_result(std::string const& name) {
    const PrimitiveParam& param = safe_at(m_inputPrims, name, "prim param");
    return param.result;
}

bool NodeImpl::add_input_prim_param(ParamPrimitive param) {
    if (m_inputPrims.find(param.name) != m_inputPrims.end()) {
        return false;
    }
    PrimitiveParam sparam;
    sparam.bInput = true;
    sparam.control = param.control;
    sparam.defl = param.defl;
    convertToEditVar(sparam.defl, param.type);
    sparam.m_wpNode = this;
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

bool NodeImpl::add_input_obj_param(ParamObject param) {
    if (m_inputObjs.find(param.name) != m_inputObjs.end()) {
        return false;
    }
    ObjectParam sparam;
    sparam.bInput = true;
    sparam.name = param.name;
    sparam.type = param.type;
    sparam.socketType = Socket_Clone;// param.socketType;
    sparam.m_wpNode = this;
    sparam.wildCardGroup = param.wildCardGroup;
    sparam.constrain = param.constrain;
    m_inputObjs.insert(std::make_pair(param.name, std::move(sparam)));
    return true;
}

bool NodeImpl::add_output_prim_param(ParamPrimitive param) {
    if (m_outputPrims.find(param.name) != m_outputPrims.end()) {
        return false;
    }
    PrimitiveParam sparam;
    sparam.bInput = false;
    sparam.control = param.control;
    sparam.defl = param.defl;
    sparam.m_wpNode = this;
    sparam.name = param.name;
    sparam.socketType = Socket_Clone;// param.socketType;
    sparam.type = param.type;
    sparam.ctrlProps = param.ctrlProps;
    sparam.wildCardGroup = param.wildCardGroup;
    sparam.bSocketVisible = param.bSocketVisible;
    sparam.constrain = param.constrain;
    m_outputPrims.insert(std::make_pair(param.name, std::move(sparam)));
    return true;
}

bool NodeImpl::add_output_obj_param(ParamObject param) {
    if (m_outputObjs.find(param.name) != m_outputObjs.end()) {
        return false;
    }
    ObjectParam sparam;
    sparam.bInput = false;
    sparam.name = param.name;
    sparam.type = param.type;
    sparam.socketType = param.socketType;
    sparam.constrain = param.constrain;
    sparam.m_wpNode = this;
    sparam.wildCardGroup = param.wildCardGroup;
    m_outputObjs.insert(std::make_pair(param.name, std::move(sparam)));
    return true;
}

void NodeImpl::init_object_link(bool bInput, const std::string& paramname, std::shared_ptr<ObjectLink> spLink, const std::string& targetParam) {
    auto iter = bInput ? m_inputObjs.find(paramname) : m_outputObjs.find(paramname);
    if (bInput)
        spLink->toparam = &iter->second;
    else
        spLink->fromparam = &iter->second;
    spLink->targetParam = targetParam;
    iter->second.links.emplace_back(spLink);
}

void NodeImpl::init_primitive_link(bool bInput, const std::string& paramname, std::shared_ptr<PrimitiveLink> spLink, const std::string& targetParam) {
    auto iter = bInput ? m_inputPrims.find(paramname) : m_outputPrims.find(paramname);
    if (bInput)
        spLink->toparam = &iter->second;
    else
        spLink->fromparam = &iter->second;
    spLink->targetParam = targetParam;
    iter->second.links.emplace_back(spLink);
}

bool NodeImpl::isPrimitiveType(bool bInput, const std::string& param_name, bool& bExist) {
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

std::vector<EdgeInfo> NodeImpl::getLinks() const {
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

std::vector<EdgeInfo> NodeImpl::getLinksByParam(bool bInput, const std::string& param_name) const {
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

bool NodeImpl::updateLinkKey(bool bInput, const zeno::EdgeInfo& edge, const std::string& oldkey, const std::string& newkey)
{
    auto& objects = bInput ? m_inputObjs : m_outputObjs;
    auto iter = objects.find(edge.inParam);
    if (iter != objects.end()) {
        for (auto spLink : iter->second.links) {
            if (auto fromParam = spLink->fromparam) {
                if (auto outnode = fromParam->m_wpNode) {
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

bool NodeImpl::moveUpLinkKey(bool bInput, const std::string& param_name, const std::string& key)
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

bool NodeImpl::removeLink(bool bInput, const EdgeInfo& edge) {
    //现有规则不允许输入对象和输入数值参数同名，所以不需要bObjLink，而且bObjLink好像只是用于给ui区分而已
    if (bInput) {
        auto iter = m_inputObjs.find(edge.inParam);
        if (iter != m_inputObjs.end()) {
            for (auto spLink : iter->second.links) {
                if (auto outNode = spLink->fromparam->m_wpNode) {
                    if (outNode->get_name() == edge.outNode && spLink->fromparam->name == edge.outParam && spLink->fromkey == edge.outKey) {
                        iter->second.links.remove(spLink);
                        return true;
                    }
                }
            }
        }
        auto iter2 = m_inputPrims.find(edge.inParam);
        if (iter2 != m_inputPrims.end()) {
            for (auto spLink : iter2->second.links) {
                if (auto outNode = spLink->fromparam->m_wpNode) {
                    if (outNode->get_name() == edge.outNode && spLink->fromparam->name == edge.outParam) {
                        iter2->second.links.remove(spLink);
                        return true;
                    }
                }
            }
        }
    }
    else {
        auto iter = m_outputObjs.find(edge.outParam);
        if (iter != m_outputObjs.end())
        {
            for (auto spLink : iter->second.links) {
                if (auto inNode = spLink->toparam->m_wpNode) {
                    if (inNode->get_name() == edge.inNode && spLink->toparam->name == edge.inParam/* && spLink->tokey == edge.inKey*/) {
                        iter->second.links.remove(spLink);
                        return true;
                    }
                }
            }
        }

        auto iter2 = m_outputPrims.find(edge.outParam);
        if (iter2 != m_outputPrims.end())
        {
            for (auto spLink : iter2->second.links) {
                if (auto inNode = spLink->toparam->m_wpNode) {
                    if (inNode->get_name() == edge.inNode && spLink->toparam->name == edge.inParam) {
                        iter2->second.links.remove(spLink);
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

std::string NodeImpl::get_viewobject_output_param() const {
    //现在暂时还没有什么标识符用于指定哪个输出口是对应输出view obj的
    //一般都是默认第一个输出obj，暂时这么规定，后续可能用标识符。
    if (m_outputObjs.empty())
        return "";
    return m_outputObjs.begin()->second.name;
}

NodeData NodeImpl::exportInfo() const
{
    NodeData node;
    node.cls = m_nodecls;
    node.name = m_name;
    node.bView = m_bView;
    node.uipos = m_pos;
    node.bnocache = m_nocache;
    //TODO: node type
    if (node.subgraph.has_value())
        node.type = Node_SubgraphNode;
    else
        node.type = Node_Normal;
    node.bLocked = false;

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

bool NodeImpl::update_param(const std::string& param, zeno::reflect::Any new_value) {
    CORE_API_BATCH
    zeno::reflect::Any old_value;
    bool ret = update_param_impl(param, new_value, old_value);
    if (ret) {
        CALLBACK_NOTIFY(update_param, param, old_value, m_inputPrims[param].defl)
        mark_dirty(true);
    }
    return ret;
}

bool NodeImpl::update_param_impl(const std::string& param, zeno::reflect::Any new_value, zeno::reflect::Any& old_value)
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

        Graph* spGraph = m_pGraph;
        assert(spGraph);

        spGraph->onNodeParamUpdated(&spParam, old_value, new_value);
        initReferLinks(&spParam);
        checkParamsConstrain();
        return true;
    }
    return false;
}

bool NodeImpl::update_param_wildcard(const std::string& param, bool isWildcard)
{
    auto& spParam = safe_at(m_inputObjs, param, "miss input param `" + param + "` on node `" + m_name + "`");
    if (isWildcard != spParam.bWildcard) {
        spParam.bWildcard = isWildcard;
        CALLBACK_NOTIFY(update_param_wildcard, param, isWildcard)
        return true;
    }
    return false;
}

bool NodeImpl::update_param_socket_type(const std::string& param, SocketType type)
{
    CORE_API_BATCH
    auto& spParam = safe_at(m_inputObjs, param, "miss input param `" + param + "` on node `" + m_name + "`");
    if (type != spParam.socketType)
    {
        spParam.socketType = type;
#if 0
        if (type == Socket_Owning)
        {
            auto spGraph = graph;
            spGraph->removeLinks(m_name, true, param);
        }
#endif
        mark_dirty(true);
        CALLBACK_NOTIFY(update_param_socket_type, param, type)
        return true;
    }
    return false;
}

bool zeno::NodeImpl::update_param_type(const std::string& param, bool bPrim, bool bInput, ParamType type)
{
    CORE_API_BATCH
        if (bPrim)
        {
            auto& prims = bInput ? m_inputPrims : m_outputPrims;
            auto prim = prims.find(param);
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
            auto object = objects.find(param);
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

bool zeno::NodeImpl::update_param_control(const std::string& param, ParamControl control)
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

bool zeno::NodeImpl::update_param_control_prop(const std::string& param, zeno::reflect::Any props)
{
    CORE_API_BATCH
    auto& spParam = safe_at(m_inputPrims, param, "miss input param `" + param + "` on node `" + m_name + "`");
    spParam.ctrlProps = props;
        CALLBACK_NOTIFY(update_param_control_prop, param, props)
        return true;
}

bool NodeImpl::update_param_visible(const std::string& name, bool bOn, bool bInput) {
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

void NodeImpl::checkParamsConstrain() {
    //ZfxContext
    auto& funcMgr = zeno::getSession().funcManager;
    ZfxContext ctx;
    ctx.spNode = this;
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

bool NodeImpl::update_param_enable(const std::string& name, bool bOn, bool bInput) {
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

bool zeno::NodeImpl::update_param_socket_visible(const std::string& param, bool bVisible, bool bInput)
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

void NodeImpl::update_param_color(const std::string& name, std::string& clr)
{
    CORE_API_BATCH
    CALLBACK_NOTIFY(update_param_color, name, clr)
}

void NodeImpl::update_layout(params_change_info& changes)
{
    CALLBACK_NOTIFY(update_layout, changes);
}

bool NodeImpl::is_loaded() const {
    return m_pNode != nullptr;
}

void NodeImpl::update_load_info(bool bDisable) {
    if (bDisable) {
        m_pNode.reset();
    }
    else {
        //重新加载，需要拿到INode的定义
        auto iter = zeno::getSession().nodeClasses.find(m_nodecls);
        if (iter == zeno::getSession().nodeClasses.end()) {
            throw makeNodeError<UnimplError>(get_path(), "cannot load the nodeclass definition");
        }
        const std::unique_ptr<INodeClass>& defcls = iter->second;
        m_pNode = defcls->new_coreinst();
    }
    CALLBACK_NOTIFY(update_load_info, bDisable);
}

params_change_info NodeImpl::update_editparams(const ParamsUpdateInfo& params, bool bSubnetInit)
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

    for (auto _pair : params) {
        if (const auto& pParam = std::get_if<ParamObject>(&_pair.param))
        {
            const ParamObject& param = *pParam;
            const std::string oldname = _pair.oldName;
            const std::string newname = param.name;

            auto& self_obj_params = param.bInput ? m_inputObjs : m_outputObjs;
            auto& new_params = param.bInput ? changes.new_inputs : changes.new_outputs;
            auto& remove_params = param.bInput ? changes.remove_inputs : changes.remove_outputs;
            auto& rename_params = param.bInput ? changes.rename_inputs : changes.rename_outputs;

            if (oldname.empty()) {
                //new added name.
                if (self_obj_params.find(newname) != self_obj_params.end()) {
                    // the new name happen to have the same name with the old name, but they are not the same param.
                    self_obj_params.erase(newname);
                    if (param.bInput)
                        obj_inputs_old.erase(newname);
                    else
                        obj_outputs_old.erase(newname);

                    remove_params.insert(newname);
                }

                ObjectParam sparam;
                sparam.name = newname;
                sparam.type = param.type;
                sparam.bWildcard = param.bWildcard;
                if (!param.bInput) {
                    sparam.socketType = Socket_Output;
                }
                else {
                    sparam.socketType = param.socketType;
                }
                sparam.m_wpNode = this;
                self_obj_params[newname] = std::move(sparam);

                new_params.insert(newname);
            }
            else if (self_obj_params.find(oldname) != self_obj_params.end()) {
                if (oldname != newname) {
                    //exist name changed.
                    self_obj_params[newname] = std::move(self_obj_params[oldname]);
                    self_obj_params.erase(oldname);

                    rename_params.insert({ oldname, newname });
                }
                else {
                    //name stays.
                }

                if (param.bInput)
                    obj_inputs_old.erase(oldname);
                else
                    obj_outputs_old.erase(oldname);

                auto& spParam = self_obj_params[newname];
                spParam.type = param.type;
                spParam.name = newname;
                spParam.bWildcard = param.bWildcard;
                if (param.bInput)
                {
                    update_param_socket_type(spParam.name, param.socketType);
                }
            }
            else {
                throw makeNodeError<KeyError>(get_path(), oldname, "the name does not exist on the node");
            }
        }
        else if (const auto& pParam = std::get_if<ParamPrimitive>(&_pair.param))
        {
            const ParamPrimitive& param = *pParam;
            const std::string oldname = _pair.oldName;
            const std::string newname = param.name;

            auto& self_prim_params = param.bInput ? m_inputPrims : m_outputPrims;
            auto& new_params = param.bInput ? changes.new_inputs : changes.new_outputs;
            auto& remove_params = param.bInput ? changes.remove_inputs : changes.remove_outputs;
            auto& rename_params = param.bInput ? changes.rename_inputs : changes.rename_outputs;

            if (oldname.empty()) {
                //new added name.
                if (self_prim_params.find(newname) != self_prim_params.end()) {
                    // the new name happen to have the same name with the old name, but they are not the same param.
                    self_prim_params.erase(newname);
                    if (param.bInput)
                        inputs_old.erase(newname);
                    else
                        outputs_old.erase(newname);

                    remove_params.insert(newname);
                }

                PrimitiveParam sparam;
                sparam.defl = param.defl;
                if (param.type == sparam.type) {
                    sparam.defl = initAnyDeflValue(sparam.type);
                }
                convertToEditVar(sparam.defl, param.type);
                sparam.name = newname;
                sparam.type = param.type;
                sparam.control = param.control;
                sparam.ctrlProps = param.ctrlProps;
                sparam.socketType = Socket_Primitve;
                sparam.bWildcard = param.bWildcard;
                sparam.m_wpNode = this;
                sparam.bSocketVisible = param.bSocketVisible;
                self_prim_params[newname] = std::move(sparam);

                new_params.insert(newname);
            }
            else if (self_prim_params.find(oldname) != self_prim_params.end()) {
                if (oldname != newname) {
                    //exist name changed.
                    self_prim_params[newname] = std::move(self_prim_params[oldname]);
                    self_prim_params.erase(oldname);

                    rename_params.insert({ oldname, newname });
                }
                else {
                    //name stays.
                }

                if (param.bInput)
                    inputs_old.erase(oldname);
                else
                    outputs_old.erase(oldname);

                auto& spParam = self_prim_params[newname];
                spParam.defl = param.defl;
                if (param.type != spParam.type) {
                    spParam.defl = initAnyDeflValue(spParam.type);
                }
                convertToEditVar(spParam.defl, spParam.type);

                spParam.name = newname;
                spParam.socketType = Socket_Primitve;
                spParam.bWildcard = param.bWildcard;
                if (param.bInput)
                {
                    if (param.type != spParam.type)
                    {
                        update_param_type(spParam.name, true, param.bInput, param.type);
                        //update_param_type(spParam->name, param.type);
                        //if (auto spNode = subgraph->getNode(oldname))
                        //    spNode->update_param_type("port", param.type);
                    }
                    update_param_control(spParam.name, param.control);
                    update_param_control_prop(spParam.name, param.ctrlProps);
                }
            }
            else {
                throw makeNodeError<KeyError>(get_path(), oldname, "the name does not exist on the node");
            }
        }
    }

    Graph* spGraph = m_pGraph;

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
    return changes;
}

void NodeImpl::trigger_update_params(const std::string& param, bool changed, params_change_info changes)
{
    if (changed)
        update_layout(changes);
}

void NodeImpl::init(const NodeData& dat)
{
    //IO init
    if (!dat.name.empty())
        m_name = dat.name;

    if (m_name == "FormSceneTree2_2") {
        int j;
        j = -0;
    }

    m_pos = dat.uipos;
    m_bView = dat.bView;
    m_bypass = dat.bypass;
    m_nocache = dat.bnocache;
    if (m_bView) {
        Graph* spGraph = m_pGraph;
        assert(spGraph);
        spGraph->viewNodeUpdated(m_name, m_bView);
    }
    if (SubnetNode* pSubnetNode = dynamic_cast<SubnetNode*>(this))
    {
        zeno::NodeType nodetype = pSubnetNode->nodeType();
        if (nodetype != zeno::Node_AssetInstance && nodetype != zeno::Node_AssetReference) {//asset初始化时已设置过customui
            pSubnetNode->setCustomUi(dat.customUi);
        }
    }
    initParams(dat);
    m_dirty = true;
}

void NodeImpl::initParams(const NodeData& dat)
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
                if (!sparam.defl.has_value()) {
                    sparam.defl = initAnyDeflValue(param.type);
                }

                // 普通子图的控件及参数类型，是由定义处决定的，而非IO值。
                //sparam.control = param.control;
                //sparam.ctrlProps = param.ctrlProps;
                //sparam.type = param.type;
                sparam.bSocketVisible = param.bSocketVisible;

                //graph记录$F相关节点
                if (Graph* spGraph = m_pGraph)
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

ShaderData NodeImpl::get_input_shader(const std::string& param, zeno::reflect::Any defl) {
    //这个函数其实就是特供ShaderFinalize
    auto iter = m_inputPrims.find(param);
    if (iter == m_inputPrims.end()) {
        throw makeNodeError<UnimplError>(get_path(), "not valid param");
    }
    ShaderData shader;

    if (!iter->second.result.has_value()) {
        bool bSucceed = false;
        shader.data = AnyToNumeric(defl, bSucceed);
        if (!bSucceed) {
            throw makeNodeError<UnimplError>(get_path(), "cannot get NumericValue on defl value");
        }
        return shader;
    }

    const Any& result = iter->second.result;
    if (result.type().hash_code() == gParamType_Shader) {
        shader = any_cast<ShaderData>(result);
    }
    else {
        bool bSucceed = false;
        shader.data = AnyToNumeric(result, bSucceed);
        if (!bSucceed) {
            throw makeNodeError<UnimplError>(get_path(), "cannot get NumericValue");
        }
    }

    ParamPath uuidpath = this->get_uuid_path() + "/" + param;
    shader.curr_param = uuidpath;
    return shader;
}

ParamType NodeImpl::get_anyparam_type(bool bInput, const std::string& name) {
    if (bInput) {
        if (m_inputObjs.find(name) != m_inputObjs.end()) {
            return m_inputObjs[name].type;
        }
        else if (m_inputPrims.find(name) != m_inputPrims.end()) {
            return m_inputPrims[name].type;
        }
    }
    else {
        if (m_outputObjs.find(name) != m_outputObjs.end()) {
            return m_outputObjs[name].type;
        }
        else if (m_outputPrims.find(name) != m_outputPrims.end()) {
            return m_outputPrims[name].type;
        }
    }
    return Param_Null;
}

bool NodeImpl::has_link_input(std::string const& id) const {
    //这个对应的是老版本的has_input
    auto iter = m_inputObjs.find(id);
    if (iter != m_inputObjs.end()) {
        return !iter->second.links.empty();
    }
    else {
        auto iter = m_inputPrims.find(id);
        if (iter != m_inputPrims.end()) {
            return !iter->second.links.empty();
        }
        return false;
    }
}

bool NodeImpl::has_input(std::string const &id) const {
    //这个has_input在旧的语义里，代表的是input obj，如果有一些边没有连上，或者有些参数没有设置默认值，就不会加到这个`input`里
    // 设计上过于随意，参考老版本的NewFBXSceneInfo.frameid
   
    // 由于新版本已经和旧版本不一致，如果要最大限度兼容，只能考虑：
    //1. 有边连着都算
    //2. 只要参数有数值结果，都算（如果是老版本的不带默认参数的，请用has_link_input）

    auto iter = m_inputObjs.find(id);
    if (iter != m_inputObjs.end()) {
        return !iter->second.links.empty();
    }
    else {
        auto iter = m_inputPrims.find(id);
        if (iter != m_inputPrims.end()) {
            const auto& _prim = iter->second;
            if (_prim.result.has_value())
                return true;
            //看有没有边连着
            return !iter->second.links.empty();
        }
        return false;
    }
}

IObject* NodeImpl::get_input_obj(std::string const& id) const {
    auto iter2 = m_inputObjs.find(id);
    if (iter2 != m_inputObjs.end()) {
        return iter2->second.spObject.get();
    }
    return nullptr;
}

/*
IObject* NodeImpl::get_input(std::string const& id) const {
    auto iter = m_inputPrims.find(id);
    if (iter != m_inputPrims.end()) {
        auto& val = iter->second.result;
        if (!val.has_value()) {
            throw makeNodeError<UnimplError>(get_path(), "cannot get prim result of " + id + "`");
        }
        const ParamType paramType = iter->second.type;
        switch (paramType) {
        case zeno::types::gParamType_Int:
        case zeno::types::gParamType_Float:
        case zeno::types::gParamType_Bool:
        case zeno::types::gParamType_Vec2f:
        case zeno::types::gParamType_Vec2i:
        case zeno::types::gParamType_Vec3f:
        case zeno::types::gParamType_Vec3i:
        case zeno::types::gParamType_Vec4f:
        case zeno::types::gParamType_Vec4i:
        case gParamType_AnyNumeric:
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
            else if (paramType == gParamType_AnyNumeric &&
                (anyType == zeno::reflect::type_info<const char*>() ||
                    anyType == zeno::reflect::type_info<zeno::String>() ||
                    anyType == zeno::reflect::type_info<std::string>())) {
                std::string str = zeno::any_cast_to_string(val);
                return std::make_unique<StringObject>(str);
            }
            else
            {
                //throw makeError<TypeError>(typeid(T));
                //error, throw expection.
            }
            return spNum;
        }
        case zeno::types::gParamType_Matrix3:
        {
            std::shared_ptr<zeno::MatrixObject> matrixObj = std::make_shared<MatrixObject>();
            matrixObj->m = zeno::reflect::any_cast<glm::mat3>(val);
            return matrixObj;
        }
        case zeno::types::gParamType_Matrix4:
        {
            std::shared_ptr<zeno::MatrixObject> matrixObj = std::make_shared<MatrixObject>();
            matrixObj->m = zeno::reflect::any_cast<glm::mat4>(val);
            return matrixObj;
        }
        case zeno::types::gParamType_String:
        {
            std::string str = zeno::any_cast_to_string(val);
            return std::make_shared<StringObject>(str);
        }
        case zeno::types::gParamType_Shader:
        {
            throw makeNodeError<UnimplError>(get_path(), "ShaderObject has been deprecated, you can get it by get_param_result, cast it into the type `ShaderData`");
        }
        case gParamType_Heatmap:
        {
            std::shared_ptr<zeno::HeatmapObject> heatmapObj = std::make_shared<zeno::HeatmapObject>();
            heatmapObj->colors = zeno::reflect::any_cast<zeno::HeatmapData>(val).colors;
            return heatmapObj;
        }
        default:
            return nullptr;
        }
    }
    else {
        auto iter2 = m_inputObjs.find(id);
        if (iter2 != m_inputObjs.end()) {
            return iter2->second.spObject;
        }
        throw makeNodeError<KeyError>(get_path(), id, "get_input");
    }
    return nullptr;
}
*/

zany NodeImpl::clone_input(std::string const &id) const {
    auto iter = m_inputPrims.find(id);
    if (iter != m_inputPrims.end()) {
        auto& val = iter->second.result;
        if (!val.has_value()) {
            throw makeNodeError<UnimplError>(get_path(), "cannot get prim result of " + id + "`");
        }
        const ParamType paramType = iter->second.type;
        switch (paramType) {
            case zeno::types::gParamType_Int:
            case zeno::types::gParamType_Float:
            case zeno::types::gParamType_Bool:
            case zeno::types::gParamType_Vec2f:
            case zeno::types::gParamType_Vec2i:
            case zeno::types::gParamType_Vec3f:
            case zeno::types::gParamType_Vec3i:
            case zeno::types::gParamType_Vec4f:
            case zeno::types::gParamType_Vec4i:
            case gParamType_AnyNumeric:
            {
                //依然有很多节点用了NumericObject，为了兼容，需要套一层NumericObject出去。
                auto spNum = std::make_unique<NumericObject>();
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
                else if (paramType == gParamType_AnyNumeric && 
                    (anyType == zeno::reflect::type_info<const char*>() ||
                     anyType == zeno::reflect::type_info<zeno::String>() ||
                     anyType == zeno::reflect::type_info<std::string>())) {
                    std::string str = zeno::any_cast_to_string(val);
                    return std::make_unique<StringObject>(str);
                }
                else
                {
                    //throw makeError<TypeError>(typeid(T));
                    //error, throw expection.
                }
                return spNum;
            }
            case zeno::types::gParamType_Matrix3:
            {
                auto matrixObj = std::make_unique<MatrixObject>();
                matrixObj->m = zeno::reflect::any_cast<glm::mat3>(val);
                return matrixObj;
            }
            case zeno::types::gParamType_Matrix4:
            {
                auto matrixObj = std::make_unique<MatrixObject>();
                matrixObj->m = zeno::reflect::any_cast<glm::mat4>(val);
                return matrixObj;
            }
            case zeno::types::gParamType_String:
            {
                std::string str = zeno::any_cast_to_string(val);
                return std::make_unique<StringObject>(str);
            }
            case zeno::types::gParamType_Shader:
            {
                throw makeNodeError<UnimplError>(get_path(), "ShaderObject has been deprecated, you can get it by get_param_result, cast it into the type `ShaderData`");
            }
            case gParamType_Heatmap:
            {
                auto heatmapObj = std::make_unique<zeno::HeatmapObject>();
                heatmapObj->colors = zeno::reflect::any_cast<zeno::HeatmapData>(val).colors;
                return heatmapObj;
            }
            default:
                return nullptr;
        }
    }
    else {
        auto iter2 = m_inputObjs.find(id);
        if (iter2 != m_inputObjs.end()) {
            return iter2->second.spObject->clone();
        }
        throw makeNodeError<KeyError>(get_path(), id, "get_input");
    }
    return nullptr;
}

void NodeImpl::set_pos(std::pair<float, float> pos) {
    m_pos = pos;
    CALLBACK_NOTIFY(set_pos, m_pos)
}

std::pair<float, float> NodeImpl::get_pos() const {
    return m_pos;
}

bool NodeImpl::in_asset_file() const {
    Graph* spGraph = m_pGraph;
    assert(spGraph);
    return getSession().assets->isAssetGraph(spGraph);
}

bool NodeImpl::set_primitive_input(std::string const& id, const zeno::reflect::Any& val) {
    auto iter = m_inputPrims.find(id);
    if (iter == m_inputPrims.end())
        return false;
    iter->second.result = val;
    return true;
}

bool NodeImpl::set_primitive_output(std::string const& id, const zeno::reflect::Any& val) {
    auto iter = m_outputPrims.find(id);
    assert(iter != m_outputPrims.end());
    if (iter == m_outputPrims.end())
        return false;
    iter->second.result = val;
    return true;
}

bool NodeImpl::set_output(std::string const& param, zany&& obj) {
    //只给旧节点模块使用，如果函数暴露reflect::Any，就会迫使所有使用这个函数的cpp文件include headers
    //会增加程序体积以及编译时间，待后续生成文件优化后再考虑处理。
    auto iter = m_outputObjs.find(param);
    if (iter != m_outputObjs.end()) {
        iter->second.spObject = std::move(obj);
        return true;
    }
    else {
        auto iter2 = m_outputPrims.find(param);
        if (iter2 != m_outputPrims.end()) {
            //兼容以前NumericObject的情况
            if (auto numObject = dynamic_cast<NumericObject*>(obj.get())) {
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
            else if (auto strObject = dynamic_cast<StringObject*>(obj.get())) {
                const auto& val = strObject->value;
                iter2->second.result = val;
            }
            return true;
        }
    }
    return false;
}

IObject* NodeImpl::get_default_output_object() {
    if (m_nodecls == "SubOutput") {
        return get_input_obj("port");
    }
    if (m_outputObjs.empty())
        return nullptr;
    return m_outputObjs.begin()->second.spObject.get();
}

zany NodeImpl::clone_default_output_object() {
    std::lock_guard lock(m_mutex);
    if (IObject* default_output_obj = get_default_output_object()) {
        return default_output_obj->clone();
    }
    return nullptr;
}

IObject* NodeImpl::get_output_obj(std::string const& param) {
    auto& spParam = safe_at(m_outputObjs, param, "miss output param `" + param + "` on node `" + m_name + "`");
    return spParam.spObject.get();
}

bool NodeImpl::checkAllOutputLinkTraced() {
    bool bAllOutputTaken = true;
    //检查是否当前节点是否所有边都被trace了
    for (const auto& [_, outparam] : m_outputObjs) {
        for (auto link : outparam.links) {
            if (!link->bTraceAndTaken) {
                bAllOutputTaken = false;
                break;
            }
        }
    }
    for (const auto& [_, outparam] : m_outputPrims) {
        for (auto link : outparam.links) {
            if (!link->bTraceAndTaken) {
                bAllOutputTaken = false;
                break;
            }
        }
    }
    return bAllOutputTaken;
}

zeno::reflect::Any NodeImpl::takeOutputPrim(PrimitiveParam* out_param, PrimitiveParam* in_param, bool& bAllOutputTaken) {
    for (auto link : out_param->links) {
        if (link->toparam == in_param) {
            link->bTraceAndTaken = true;
            break;
        }
    }
    bAllOutputTaken = checkAllOutputLinkTraced();
    return out_param->result;
}

zeno::reflect::Any NodeImpl::takeOutputPrim(const std::string& out_param, const std::string& in_param, bool& bAllOutputTaken) {
    auto iter = m_outputPrims.find(out_param);
    if (iter == m_outputPrims.end())
        throw makeNodeError<KeyError>(get_path(), out_param, "no such output prim param");

    auto& outparam = iter->second;
    for (auto link : outparam.links) {
        if (link->toparam->name == in_param) {
            link->bTraceAndTaken = true;
            break;
        }
    }
    bAllOutputTaken = checkAllOutputLinkTraced();
    return outparam.result;
}

void NodeImpl::mark_takeover() {
    m_takenover = true;
    clearCalcResults();
    mark_dirty(false);
    reportStatus(false, Node_RunSucceed);
}

bool NodeImpl::is_takenover() const {
    return m_takenover;
}

zany NodeImpl::takeOutputObject(ObjectParam* out_param, ObjectParam* in_param, bool& bAllOutputTaken) {
    for (auto link : out_param->links) {
        if (link->toparam == in_param) {
            link->bTraceAndTaken = true;
            break;
        }
    }
    bAllOutputTaken = checkAllOutputLinkTraced();
    return std::move(out_param->spObject);
}

zany NodeImpl::takeOutputObject(const std::string& out_param, const std::string& in_param, bool& bAllOutputTaken) {
    auto iter = m_outputObjs.find(out_param);
    if (iter == m_outputObjs.end())
        throw makeNodeError<KeyError>(get_path(), out_param, "no such output object param");
    auto& outparam = iter->second;
    for (auto link : outparam.links) {
        if (link->toparam->name == in_param) {
            link->bTraceAndTaken = true;
            break;
        }
    }
    bAllOutputTaken = checkAllOutputLinkTraced();
    return std::move(outparam.spObject);
}

std::vector<zany> NodeImpl::get_output_objs() {
    std::vector<zany> objs;
    for (const auto& [name, objparam] : m_outputObjs) {
        objs.push_back(objparam.spObject->clone());
    }
    return objs;
}

zfxvariant NodeImpl::execute_fmla(const std::string& expression)
{
    std::string code = expression;

    ZfxContext ctx;
    ctx.spNode = this;
    ctx.spObject = nullptr;
    ctx.code = code;
    ctx.runover = ATTR_GEO;
    ctx.bSingleFmla = true;
    if (!ctx.code.empty() && ctx.code.back() != ';') {
        ctx.code.push_back(';');
    }

    ZfxExecute zfx(ctx.code, &ctx);
    zeno::zfxvariant res = zfx.execute_fmla();
    return res;
}

float NodeImpl::resolve(const std::string& expression, const ParamType type) {
    const zfxvariant& res = execute_fmla(expression);
    if (std::holds_alternative<int>(res)) {
        return std::get<int>(res);
    }
    else if (std::holds_alternative<float>(res)) {
        return std::get<float>(res);
    }
    else {
        throw makeNodeError<UnimplError>(get_path(), "the result of formula is not numeric");
    }
    //TODO: kframe issues
    //k帧太麻烦，现阶段用不上先不处理
}

std::string NodeImpl::resolve_string(const std::string& fmla, const std::string& defl) {
    try
    {
        const zfxvariant& res = execute_fmla(fmla);
        if (std::holds_alternative<std::string>(res)) {
            return std::get<std::string>(res);
        }
    }
    catch(...)
    {
    }
    return defl;
}

void NodeImpl::initTypeBase(zeno::reflect::TypeBase* pTypeBase)
{
    m_pTypebase = pTypeBase;
}

bool NodeImpl::is_continue_to_run() {
    return false;
}

void NodeImpl::increment() {

}

void NodeImpl::reset_forloop_settings() {

}

std::shared_ptr<IObject> NodeImpl::get_iterate_object() {
    return nullptr;
}

std::vector<std::pair<std::string, bool>> zeno::NodeImpl::getWildCardParams(const std::string& param_name, bool bPrim)
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

void zeno::NodeImpl::getParamTypeAndSocketType(const std::string& param_name, bool bPrim, bool bInput, ParamType& paramType, SocketType& socketType, bool& bWildcard)
{
    if (bPrim) {
        auto iter = bInput ? m_inputPrims.find(param_name) : m_outputPrims.find(param_name);
        if (bInput ? (iter != m_inputPrims.end()) : (iter != m_outputPrims.end())) {
            paramType = iter->second.type;
            socketType = iter->second.socketType;
            bWildcard = iter->second.bWildcard;
            return;
    }
    }
    else {
        auto iter = bInput ? m_inputObjs.find(param_name) : m_outputObjs.find(param_name);
        if (bInput ? (iter != m_inputObjs.end()) : (iter != m_outputObjs.end())) {
            paramType = iter->second.type;
            socketType = iter->second.socketType;
            bWildcard = iter->second.bWildcard;
            return;
        }
    }
    paramType = Param_Null;
    socketType = Socket_Primitve;
}

template<class T, class E> T NodeImpl::resolveVec(const zeno::reflect::Any& defl, const ParamType type)
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
