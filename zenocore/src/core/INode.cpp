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
#include <zeno/core/IParam.h>
#include <zeno/DictObject.h>
#include <zeno/ListObject.h>
#include <zeno/utils/helper.h>
#include <zeno/utils/uuid.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/GraphException.h>
#include <zeno/formula/formula.h>
#include <zeno/core/ReferManager.h>


namespace zeno {

ZENO_API INode::INode() {}

void INode::initUuid(std::shared_ptr<Graph> pGraph, const std::string nodecls) {
    m_nodecls = nodecls;
    this->graph = pGraph;

    m_uuid = generateUUID(nodecls);
    std::list<std::string> path;
    path.push_front(m_uuid);
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
            path.push_front(pSubnetNode->m_uuid);
            pGraph = pSubnetNode->graph.lock();
        }
    }
    m_uuidPath = ObjPath(path);
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

ZENO_API zvariant INode::get_input_defl(std::string const& name)
{
    std::shared_ptr<IParam> param = get_input_param(name);
    return param->defl;
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

ZENO_API ObjPath INode::get_path() const {
    std::list<std::string> path;
    path.push_front(m_name);

    std::shared_ptr<Graph> pGraph = graph.lock();

    while (pGraph) {
        const std::string name = pGraph->getName();
        if (name == "main") {
            path.push_front("main");
            break;
        }
        else {
            if (!pGraph->optParentSubgNode.has_value())
                break;
            auto pSubnetNode = pGraph->optParentSubgNode.value();
            assert(pSubnetNode);
            path.push_front(pSubnetNode->m_name);
            pGraph = pSubnetNode->graph.lock();
        }
    }
    return ObjPath(path);
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
    for (const auto& [name, param] : m_inputs) {
        for (const auto& link : param->links) {
            if (link->lnkProp == Link_Ref) {
                auto spOutParam = link->fromparam.lock();
                auto spPreviusNode = spOutParam->m_wpNode.lock();
                spPreviusNode->mark_previous_ref_dirty();
            }
        }
    }
}

void INode::onInterrupted() {
    mark_dirty(true);
    mark_previous_ref_dirty();
}

ZENO_API void INode::mark_dirty(bool bOn, bool bWholeSubnet)
{
    scope_exit sp([&] {
        m_status = Node_DirtyReadyToRun;  //修改了数据，标脏，并置为此状态。（后续在计算过程中不允许修改数据，所以markDirty理论上是前端驱动）
        reportStatus(m_dirty, m_status);
    });

    if (m_dirty == bOn)
        return;

    m_dirty = bOn;
    if (m_dirty) {
        for (auto& [name, param] : m_outputs) {
            for (auto link : param->links) {
                auto inParam = link->toparam.lock();
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
    for (auto const& [name, param] : m_outputs)
    {
        if (auto spObj = std::dynamic_pointer_cast<IObject>(param->result)) {
            if (spObj->key().empty()) {
                continue;
            }
            getSession().objsMan->collect_removing_objs(spObj->key());
        }
    }
}

ZENO_API void INode::complete() {}

ZENO_API void INode::preApply() {
    if (!m_dirty)
        return;

    reportStatus(true, Node_Pending);

    //TODO: the param order should be arranged by the descriptors.
    for (const auto& [name, param] : m_inputs) {
        bool ret = requireInput(param);
        if (!ret)
            zeno::log_warn("the param {} may not be initialized", name);
    }
}

ZENO_API void INode::registerObjToManager()
{
    for (auto const& [name, param] : m_outputs)
    {
        if (auto spObj = std::dynamic_pointer_cast<IObject>(param->result)) {

            if (std::dynamic_pointer_cast<NumericObject>(spObj) ||
                std::dynamic_pointer_cast<StringObject>(spObj)) {
                return;
            }

            if (spObj->key().empty())
            {
                //如果当前节点是引用前继节点产生的obj，则obj.key不为空，此时就必须沿用之前的id，
                //以表示“引用”，否则如果新建id，obj指针可能是同一个，会在manager引起混乱。
                spObj->update_key(m_uuid);
            }

            const std::string& key = spObj->key();
            assert(!key.empty());
            param->result->nodeId = m_name;

            auto& objsMan = getSession().objsMan;
            std::shared_ptr<INode> spNode = shared_from_this();
            objsMan->collectingObject(spObj, spNode, m_bView);
        }
    }
}

ZENO_API bool INode::requireInput(std::string const& ds) {
    auto param = get_input_param(ds);
    return requireInput(param);
}

zany INode::get_output_result(std::shared_ptr<INode> outNode, std::string out_param, bool bCopy) {
    zany outResult = outNode->get_output(out_param);
    if (bCopy && outResult) {
        outResult = outResult->clone();
        //if (outResult->key().empty()) {
        //    outResult->key = generateUUID();
        //}
    }
    return outResult;
}

ZENO_API bool INode::requireInput(std::shared_ptr<IParam> in_param) {
    if (!in_param)
        return false;

    if (in_param->links.empty()) {
        in_param->result = process(in_param);
        return true;    //旧版本的requireInput指的是是否有连线，如果想兼容旧版本，这里可以返回false，但使用量不多，所以就修改它的定义。
    }

    switch (in_param->type)
    {
        case Param_Dict:
        {
            std::shared_ptr<DictObject> spDict;
            //连接的元素是list还是list of list的规则，参照Graph::addLink下注释。
            bool bDirecyLink = false;
            const auto& inLinks = in_param->links;
            if (inLinks.size() == 1)
            {
                std::shared_ptr<ILink> spLink = inLinks.front();
                std::shared_ptr<IParam> out_param = spLink->fromparam.lock();
                std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();

                if (out_param->type == in_param->type && !spLink->tokey.empty())
                {
                    bDirecyLink = true;
                    GraphException::translated([&] {
                        outNode->doApply();
                    }, outNode.get());
                    zany outResult = get_output_result(outNode, out_param->name, Link_Copy == spLink->lnkProp);
                    spDict = std::dynamic_pointer_cast<DictObject>(outResult);
                }
            }
            if (!bDirecyLink)
            {
                spDict = std::make_shared<DictObject>();
                for (const auto& spLink : in_param->links)
                {
                    const std::string& keyName = spLink->tokey;
                    std::shared_ptr<IParam> out_param = spLink->fromparam.lock();
                    std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();

                    GraphException::translated([&] {
                        outNode->doApply();
                    }, outNode.get());

                    zany outResult = get_output_result(outNode, out_param->name, Link_Copy == spLink->lnkProp);
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
                std::shared_ptr<ILink> spLink = in_param->links.front();
                std::shared_ptr<IParam> out_param = spLink->fromparam.lock();
                std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();

                if (out_param->type == in_param->type && !spLink->tokey.empty()) {
                    bDirectLink = true;

                    GraphException::translated([&] {
                        outNode->doApply();
                    }, outNode.get());

                    zany outResult = get_output_result(outNode, out_param->name, Link_Copy == spLink->lnkProp);
                    spList = std::dynamic_pointer_cast<ListObject>(outResult);
                }
            }
            if (!bDirectLink)
            {
                auto oldinput = std::dynamic_pointer_cast<ListObject>(in_param->result);

                spList = std::make_shared<ListObject>();
                int indx = 0;
                for (const auto& spLink : in_param->links)
                {
                    //list的情况下，keyName是不是没意义，顺序怎么维持？
                    std::shared_ptr<IParam> out_param = spLink->fromparam.lock();
                    std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();
                    if (outNode->is_dirty()) {  //list中的元素是dirty的，重新计算并加入list
                        GraphException::translated([&] {
                            outNode->doApply();
                        }, outNode.get());

                        zany outResult = get_output_result(outNode, out_param->name, Link_Copy == spLink->lnkProp);
                        spList->push_back(outResult);
                        //spList->dirtyIndice.insert(indx);
                    } else {
                        zany outResult = get_output_result(outNode, out_param->name, Link_Copy == spLink->lnkProp);
                        spList->push_back(outResult);
                    }
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
                std::shared_ptr<ILink> spLink = *in_param->links.begin();
                std::shared_ptr<IParam> out_param = spLink->fromparam.lock();
                std::shared_ptr<INode> outNode = out_param->m_wpNode.lock();

                GraphException::translated([&] {
                    outNode->doApply();
                }, outNode.get());

                in_param->result = get_output_result(outNode, out_param->name, Link_Copy == spLink->lnkProp);
            }
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

    zeno::scope_exit spe([&] {//apply时根据情况将IParam标记为modified，退出时将所有IParam标记为未modified
        for (auto const& [name, param] : m_outputs)
            param->m_idModify = false;
        });

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
        apply();
    }
    log_debug("==> leave {}", m_name);

    registerObjToManager();
    reportStatus(false, Node_RunSucceed);
}

ZENO_API std::vector<std::shared_ptr<IParam>> INode::get_input_params() const
{
    std::vector<std::shared_ptr<IParam>> params;
    //TODO: 如果参数deprecated，是否还需要加入inputs_? 要
    if (m_nodecls != "DeprecatedNode") {

        for (const ParamTab& tab : nodeClass->m_customui.tabs)
        {
            for (const ParamGroup& group : tab.groups)
            {
                for (const ParamInfo& param : group.params)
                {
                    auto it = m_inputs.find(param.name);
                    if (it == m_inputs.end()) {
                        zeno::log_warn("unknown param {}", param.name);
                        continue;
                    }
                    params.push_back(it->second);
                }
            }
        }
    }
    else {
        //TODO: the order of deprecated node.
        for (auto& [name, param] : m_inputs) {
            params.push_back(param);
        }
    }
    return params;
}

ZENO_API std::vector<std::shared_ptr<IParam>> INode::get_output_params() const
{
    std::vector<std::shared_ptr<IParam>> params;
    if (m_nodecls != "DeprecatedNode") {
        for (auto param : nodeClass->m_customui.outputs) {
            auto it = m_outputs.find(param.name);
            if (it == m_outputs.end()) {
                zeno::log_warn("unknown param {}", param.name);
                continue;
            }
            params.push_back(it->second);
        }
    }
    else {
        for (auto& [name, param] : m_outputs) {
            params.push_back(param);
        }
    }
    return params;
}

ZENO_API void INode::set_input_defl(std::string const& name, zvariant defl) {
    std::shared_ptr<IParam> param = get_input_param(name);
    param->defl = defl;
}

ZENO_API std::shared_ptr<IParam> INode::get_input_param(std::string const& param) const {
    auto it = m_inputs.find(param);
    if (it != m_inputs.end())
        return it->second;
    return nullptr;
}

void INode::add_input_param(std::shared_ptr<IParam> param) {
    m_inputs.insert(std::make_pair(param->name, param));
}

void INode::add_output_param(std::shared_ptr<IParam> param) {
    m_outputs.insert(std::make_pair(param->name, param));
}

ZENO_API std::shared_ptr<IParam> INode::get_output_param(std::string const& param) const {
    auto it = m_outputs.find(param);
    if (it != m_outputs.end())
        return it->second;
    return nullptr;
}

ZENO_API void INode::set_result(bool bInput, const std::string& name, zany spObj) {
    if (bInput) {
        auto param = safe_at(m_inputs, name, "");
        param->result = spObj;
    }
    else {
        auto param = safe_at(m_outputs, name, "");
        param->result = spObj;
    }
}

ZENO_API std::string INode::get_viewobject_output_param() const {
    //现在暂时还没有什么标识符用于指定哪个输出口是对应输出view obj的
    //一般都是默认第一个输出obj，暂时这么规定，后续可能用标识符。
    auto params = get_output_params();
    if (!params.empty())
        return params[0]->name;
    else
        return "";
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

    for (auto sparam : get_input_params()) {
        ParamInfo param;
        param.name = sparam->name;
        param.bInput = true;
        param.control = sparam->control;
        param.ctrlProps = sparam->optCtrlprops;
        param.type = sparam->type;
        param.defl = sparam->defl;
        param.socketType = sparam->socketType;
        for (auto link : sparam->links) {
            EdgeInfo info;
            auto outParam = link->fromparam.lock();
            auto outNode = outParam->m_wpNode.lock();
            info.outNode = outNode->get_name();
            info.outParam = outParam->name;
            info.outKey = link->fromkey;
            info.inNode = m_name;
            info.inParam = param.name;
            info.inKey = link->tokey;
            info.lnkfunc = link->lnkProp;
            param.links.push_back(info);
        }
        node.inputs.push_back(param);
    }

    for (auto sparam : get_output_params()) {
        ParamInfo param;
        param.bInput = false;
        param.name = sparam->name;
        param.control = sparam->control;
        param.ctrlProps = sparam->optCtrlprops;
        param.socketType = sparam->socketType;
        param.type = sparam->type;
        param.defl = sparam->defl;
        for (auto link : sparam->links) {
            EdgeInfo info;
            auto inParam = link->toparam.lock();
            auto inNode = inParam->m_wpNode.lock();
            info.inNode = inNode->get_name();
            info.inParam = inParam->name;
            info.inKey = link->tokey;
            info.outNode = m_name;
            info.outParam = param.name;
            info.outKey = link->fromkey;
            info.lnkfunc = link->lnkProp;
            param.links.push_back(info);
        }
        node.outputs.push_back(param);
    }
    return node;
}

ZENO_API bool INode::update_param(const std::string& param, const zvariant& new_value) {
    CORE_API_BATCH
    std::shared_ptr<IParam> spParam = safe_at(m_inputs, param, "miss input param `" + param + "` on node `" + m_name + "`");
    if (!zeno::isEqual(spParam->defl, new_value, spParam->type))
    {
        zvariant old_value = spParam->defl;
        spParam->defl = new_value;

        std::shared_ptr<Graph> spGraph = graph.lock();
        assert(spGraph);

        spGraph->onNodeParamUpdated(spParam, old_value, new_value);
        CALLBACK_NOTIFY(update_param, param, old_value, new_value)
        mark_dirty(true);
        getSession().referManager->checkReference(m_uuidPath, spParam->name);
        return true;
    }
    return false;
}

ZENO_API params_change_info INode::update_editparams(const ParamsUpdateInfo& params)
{
    params_change_info ret;
    return ret;
}

void INode::directly_setinputs(std::map<std::string, zany> inputs)
{
    for (auto& [name, val] : inputs) {
        std::shared_ptr<IParam> sparam = get_input_param(name);
        if (!sparam) {
            sparam = std::make_shared<IParam>();
            sparam->name = name;
            sparam->m_wpNode = shared_from_this();
            sparam->type = Param_Null;
            sparam->defl = zvariant();
        }
        sparam->result = val;
    }
}

std::map<std::string, zany> INode::getoutputs()
{
    std::map<std::string, zany> output_res;
    for (const auto& [name, param] : this->m_outputs) {
        output_res.insert(std::make_pair(name, param->result));
    }
    return output_res;
}

std::vector<std::pair<std::string, zany>> INode::getinputs()
{
    std::vector<std::pair<std::string, zany>> input_res;
    for (const auto& [name, param] : this->m_inputs) {
        input_res.push_back(std::make_pair(name, param->result));
    }
    return input_res;
}

std::pair<std::string, std::string> INode::getinputbound(std::string const& param, std::string const& msg) const
{
    std::shared_ptr<IParam> spParam = safe_at(m_inputs, param, "miss input param `" + param + "` on node `" + m_name + "`");
    if (!spParam->links.empty()) {
        auto lnk = *spParam->links.begin();
        auto outparam = lnk->fromparam.lock();
        if (outparam) {
            outparam->name;
            auto pnode = outparam->m_wpNode.lock();
            if (pnode) {
                auto id = pnode->get_ident();
                return { id, outparam->name };
            }
        }
    }
    throw makeError<KeyError>(m_name, msg);
}

std::vector<std::pair<std::string, zany>> INode::getoutputs2()
{
    std::vector<std::pair<std::string, zany>> output_res;
    for (const auto& [name, param] : this->m_outputs) {
        output_res.push_back(std::make_pair(name, param->result));
    }
    return output_res;
}

ZENO_API void INode::init(const NodeData& dat)
{
    //IO init
    if (!dat.name.empty())
        m_name = dat.name;

    m_pos = dat.uipos;
    m_bView = dat.bView;
    if (m_bView) {
        std::shared_ptr<Graph> spGraph = graph.lock();
        assert(spGraph);
        spGraph->viewNodeUpdated(m_name, m_bView);
    }
    initParams(dat);
    m_dirty = true;
}

ZENO_API void INode::initParams(const NodeData& dat)
{
    for (const ParamInfo& param : dat.inputs)
    {
        std::shared_ptr<IParam> sparam = get_input_param(param.name);
        if (!sparam) {
            //legacy zsg有大量此类参数，导致占用输出，因此先屏蔽
            //zeno::log_warn("input param `{}` is not registerd in current zeno version");
            continue;
        }
        sparam->defl = param.defl;
        sparam->control = param.control;
        sparam->optCtrlprops = param.ctrlProps;
        sparam->m_wpNode = shared_from_this();
    }
    for (const ParamInfo& param : dat.outputs)
    {
        std::shared_ptr<IParam> sparam = get_output_param(param.name);
        if (!sparam) {
            //zeno::log_warn("output param `{}` is not registerd in current zeno version");
            continue;
        }
        sparam->defl = param.defl;
        sparam->control = param.control;
        sparam->optCtrlprops = param.ctrlProps;
        sparam->m_wpNode = shared_from_this();
    }
}

ZENO_API bool INode::has_input(std::string const &id) const {
    auto param = get_input_param(id);
    return param != nullptr && param->result != nullptr;
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

ZENO_API bool INode::set_input(std::string const& param, zany obj) {
    std::shared_ptr<IParam> spParam = safe_at(m_inputs, param, "miss input param `" + param + "` on node `" + m_name + "`");
    spParam->result = obj;
    return true;
}

ZENO_API bool INode::has_output(std::string const& name) const {
    return get_output_param(name) != nullptr;
}

ZENO_API bool INode::set_output(std::string const & param, zany obj) {
    std::shared_ptr<IParam> spParam = safe_at(m_outputs, param, "miss output param `" + param + "` on node `" + m_name + "`");
    spParam->result = obj;
    return true;
}

ZENO_API zany INode::get_output(std::string const& param) {
    std::shared_ptr<IParam> spParam = safe_at(m_outputs, param, "miss output param `" + param + "` on node `" + m_name + "`");
    return spParam->result;
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
    int frame = getGlobalState()->getFrameId();
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
    //TODO: deprecated
    std::shared_ptr<Graph> spGraph = graph.lock();
    assert(spGraph);
    return TempNodeCaller(spGraph.get(), id);
}

float INode::resolve(const std::string& formulaOrKFrame, const ParamType type)
{
    int frame = getGlobalState()->getFrameId();
    if (zeno::starts_with(formulaOrKFrame, "=")) {
        std::string code = formulaOrKFrame.substr(1);
        std::set<std::string>paths = zeno::getReferPath(code);
        std::string currPath = zeno::objPathToStr(get_path());
        currPath = currPath.substr(0, currPath.find_last_of("/"));
        for (auto& path : paths)
        {
            auto absolutePath = zeno::absolutePath(currPath, path);
            if (absolutePath != path)
            {
                code.replace(code.find(path), path.size(), absolutePath);
            }
        }
        Formula fmla(code);
        float res = 0.;
        int ret = fmla.parse(res);
        return res;
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

    int frame = getGlobalState()->getFrameId();
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
        {
            if (std::holds_alternative<std::string>(defl))
            {
                result = zeno::parseCurveObj(std::get<std::string>(defl));
            }
            break;
        }
        case Param_Heatmap:
        {
            if (std::holds_alternative<std::string>(defl))
                result = zeno::parseHeatmapObj(std::get<std::string>(defl));
            break;
        }
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
