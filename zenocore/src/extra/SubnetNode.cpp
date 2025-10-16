#include <zeno/core/NodeImpl.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INodeClass.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/types/DummyObject.h>
#include <zeno/utils/log.h>
#include <zeno/core/CoreParam.h>
#include <zeno/core/Assets.h>
#include <zeno/utils/helper.h>
#include "zeno_types/reflect/reflection.generated.hpp"
#include <zeno/zeno.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/Timer.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/utils/interfaceutil.h>


namespace zeno {

SubnetNode::SubnetNode(INode* pNode)
    : NodeImpl(pNode)
    , m_subgraph(std::make_shared<Graph>(""))
    , m_bLocked(true)
    , m_bClearSubnet(false)
{
    m_subgraph->initParentSubnetNode(this);

    //auto cl = safe_at(getSession().nodeClasses, "Subnet", "node class name").get();
    //m_customUi = cl->m_customui;
    //添加一些default的输入输出
    zeno::ParamTab tab;
    zeno::ParamGroup default_group;

    zeno::ParamUpdateInfo info;

    zeno::ParamPrimitive param;
    param.bInput = true;
    param.name = "data_input";
    param.defl = zeno::reflect::make_any<zeno::PrimVar>(zeno::PrimVar(0));;
    param.type = zeno::types::gParamType_Int;
    param.socketType = zeno::Socket_Primitve;
    param.control = zeno::Lineedit;
    param.bSocketVisible = false;
    info.param = param;
    default_group.params.push_back(param);

    zeno::ParamPrimitive outputparam;
    outputparam.bInput = false;
    outputparam.name = "data_output";
    outputparam.defl = 2;
    outputparam.type = gParamType_Int;
    outputparam.socketType = zeno::Socket_Primitve;
    outputparam.bSocketVisible = false;
    info.param = outputparam;

    zeno::ParamObject objInput;
    objInput.bInput = true;
    objInput.name = "Input";
    objInput.type = gParamType_Geometry;

    zeno::ParamObject objOutput;
    objOutput.bInput = false;
    objOutput.name = "Output";
    objOutput.type = gParamType_Geometry;
    objOutput.socketType = zeno::Socket_Output;

    tab.groups.emplace_back(std::move(default_group));
    m_customUi.inputPrims.emplace_back(std::move(tab));
    m_customUi.inputObjs.push_back(objInput);
    m_customUi.outputPrims.push_back(outputparam);
    m_customUi.outputObjs.push_back(objOutput);

    m_customUi.uistyle.background = "#1D5F51";
    m_customUi.uistyle.iconResPath = ":/icons/node/subnet.svg";
}

SubnetNode::~SubnetNode() = default;

void SubnetNode::initParams(const NodeData& dat)
{
    NodeImpl::initParams(dat);
    m_bClearSubnet = dat.bclearsbn;
    if (dat.subgraph)
        m_subgraph->init(*dat.subgraph);
    if (zeno::Node_AssetInstance == nodeType()) {
        m_bLocked = dat.bLocked;
    }
}

void SubnetNode::mark_clean() {
    NodeImpl::mark_clean();
    m_subgraph->mark_clean();
}

NodeType SubnetNode::nodeType() const {
    if (isAssetsNode()) {
        if (in_asset_file())
            return zeno::Node_AssetReference;
        else
            return zeno::Node_AssetInstance;
    }
    return zeno::Node_SubgraphNode;
}

Graph* SubnetNode::get_subgraph() const
{
    return m_subgraph.get();
}

void SubnetNode::init_graph(std::shared_ptr<Graph> subg) {
    m_subgraph = subg;
}

bool SubnetNode::isAssetsNode() const {
    zeno::Asset asst = zeno::getSession().assets->getAsset(get_nodecls());
    return !asst.m_info.name.empty();
}

bool SubnetNode::is_loaded() const {
    //TODO: 资产没加载的情况
    return true;
}

bool SubnetNode::is_locked() const {
    if (nodeType() == Node_AssetInstance || nodeType() == Node_AssetReference)
        return m_bLocked;
    else
        return false;
}

void SubnetNode::set_locked(bool bLocked) {
    if (nodeType() == Node_AssetInstance) {
        m_bLocked = bLocked;
        CALLBACK_NOTIFY(lockChanged)
    }
}

bool SubnetNode::is_clearsubnet() const {
    return m_bClearSubnet;
}

void SubnetNode::set_clearsubnet(bool bOn) {
    m_bClearSubnet = bOn;
    CALLBACK_NOTIFY(clearSubnetChanged, bOn)
    mark_dirty(true);
}

params_change_info SubnetNode::update_editparams(const ParamsUpdateInfo& params, bool bSubnetInit)
{
    params_change_info changes = NodeImpl::update_editparams(params);
    //update subnetnode.
    //没有锁定的节点（包括资产实例和普通子图，都可以在这里更新Subnet的SubInput/SubOutput等）
    if (!is_locked()) {
        for (auto name : changes.new_inputs) {
            NodeImpl* newNode = m_subgraph->createNode("SubInput", name);

            if (bSubnetInit) {
                if (name == "data_input") {
                    newNode->set_pos({700, 0});
                }
                else if (name == "Input") {
                    newNode->set_pos({0, 0});
                }
            }

            //这里SubInput的类型其实是和Subnet节点创建预设的参数对应，参考AddNodeCommand

            bool exist;     //subnet通过自定义参数面板创建SubInput节点时，根据实际情况添加primitive/obj类型的port端口
            bool isprim = isPrimitiveType(true, name, exist);
            if (isprim) {
                zeno::ParamPrimitive primitive;
                primitive.bInput = false;
                primitive.name = "port";
                primitive.socketType = Socket_Output;
                newNode->add_output_prim_param(primitive);
            }
            else if (!isprim && exist) {
                zeno::ParamObject paramObj;
                paramObj.bInput = false;
                paramObj.name = "port";
                paramObj.type = get_anyparam_type(true, name);//  gParamType_Geometry;
                paramObj.socketType = Socket_Output;
                newNode->add_output_obj_param(paramObj);
            }

            for (const auto& [param, _] : params) {     //创建Subinput时,更新Subinput的port接口类型
                if (auto paramPrim = std::get_if<ParamPrimitive>(&param)) {
                    if (name == paramPrim->name) {
                        newNode->update_param_type("port", true, false, paramPrim->type);
                        break;
                    }
                }
            }
            params_change_info changes;
            changes.new_outputs.insert("port");
            changes.outputs.push_back("port");
            changes.outputs.push_back("hasValue");
            newNode->update_layout(changes);
        }
        for (const auto& [old_name, new_name] : changes.rename_inputs) {
            m_subgraph->updateNodeName(old_name, new_name);
        }
        for (auto name : changes.remove_inputs) {
            m_subgraph->removeNode(name);
        }

        for (auto name : changes.new_outputs) {
            NodeImpl* newNode = m_subgraph->createNode("SubOutput", name);

            if (bSubnetInit) {
                if (name == "data_output") {
                    newNode->set_pos({ 700, 500 });
                }
                else if (name == "Output") {
                    newNode->set_pos({ 0, 500 });
                    //newNode->set_view(true);
                }
            }

            bool exist;
            bool isprim = isPrimitiveType(false, name, exist);
            if (isprim) {
                zeno::ParamPrimitive primitive;
                primitive.bInput = true;
                primitive.name = "port";
                primitive.type = gParamType_Int;
                primitive.socketType = Socket_Primitve;
                primitive.defl = 0;
                newNode->add_input_prim_param(primitive);
            }
            else if (!isprim && exist) {
                zeno::ParamObject paramObj;
                paramObj.bInput = true;
                paramObj.name = "port";
                paramObj.type = get_anyparam_type(false, name); //gParamType_Geometry;
                paramObj.socketType = Socket_Clone;
                newNode->add_input_obj_param(paramObj);
            }
            params_change_info changes;
            changes.new_inputs.insert("port");
            changes.inputs.push_back("port");
            newNode->update_layout(changes);
        }
        for (const auto& [old_name, new_name] : changes.rename_outputs) {
            m_subgraph->updateNodeName(old_name, new_name);
        }
        for (auto name : changes.remove_outputs) {
            m_subgraph->removeNode(name);
        }
    }
    //prim的输入类型变化时，可能需要更新对应subinput节点port端口的类型
    for (auto _pair : params) {
        if (const auto& pParam = std::get_if<ParamObject>(&_pair.param)) {
            //检测类型是否变化了
            const ParamObject& param = *pParam;
            auto subionode = m_subgraph->getNode(param.name);
            bool bInput = !param.bInput;    //输入参数对应输出参数的port，反之亦然
            if (subionode) {
                ParamType old_paramtype;
                SocketType old_socketype; //DEPRECATED
                bool _wildcard; //DEPRECATED
                subionode->getParamTypeAndSocketType("port", false, bInput, old_paramtype, old_socketype, _wildcard);
                if (old_paramtype != param.type) {
                    subionode->update_param_type("port", false, bInput, param.type);
                    //TODO:为了避免连线出问题，删掉所有连线
                }
            }
        }
        else if (const auto& pParam = std::get_if<ParamPrimitive>(&_pair.param)) {
            const ParamPrimitive& param = *pParam;
            if (param.bInput && 
                changes.new_inputs.find(param.name) == changes.new_inputs.end() && 
                changes.remove_inputs.find(param.name) == changes.remove_inputs.end()) {
                auto inputnode = m_subgraph->getNode(param.name);
                if (inputnode) {
                    ParamType paramtype;
                    SocketType socketype;
                    bool _wildcard;
                    inputnode->getParamTypeAndSocketType("port", true, false, paramtype, socketype, _wildcard);
                    if (paramtype != param.type) {
                        inputnode->update_param_type("port", true, false, param.type);
                        for (auto& link : inputnode->getLinksByParam(false, "port")) {
                            if (auto linktonode = m_subgraph->getNode(link.inNode)) {
                                ParamType paramType;
                                SocketType socketType;
                                bool bWildcard;
                                linktonode->getParamTypeAndSocketType(link.inParam, true, true, paramType, socketType, bWildcard);
                                if (!outParamTypeCanConvertInParamType(param.type, paramType, Role_OutputPrimitive, Role_InputPrimitive)) {
                                    m_subgraph->removeLink(link);
                                }
                            }
                        }
                        for (auto& link : getLinksByParam(true, param.name)) {
                            if (auto spgraph = m_pGraph) {
                                if (auto linktonode = spgraph->getNode(link.outNode)) {
                                    ParamType paramType;
                                    SocketType socketType;
                                    bool bWildcard;
                                    linktonode->getParamTypeAndSocketType(link.outParam, true, false, paramType, socketType, bWildcard);
                                    if (!outParamTypeCanConvertInParamType(paramType, param.type, Role_OutputPrimitive, Role_InputPrimitive)) {
                                        spgraph->removeLink(link);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return changes;
}

void SubnetNode::mark_subnetdirty(bool bOn)
{
    if (bOn) {
        m_subgraph->markDirtyAndCleanup();
    }
}

float SubnetNode::time() const {
    if (m_subgraph)
        return m_subgraph->statistic_cpu_used();
    else
        return 0;
}

void SubnetNode::apply() {
    for (auto const &subinput_node: m_subgraph->getSubInputs()) {
        auto subinput = m_subgraph->getNode(subinput_node);
        auto iter = m_inputObjs.find(subinput_node);
        if (iter != m_inputObjs.end()) {
            //object type.
            if (iter->second.spObject) {
                //要拷贝一下才能赋值到SubInput的port参数
                zany spObject = iter->second.spObject->clone();
                spObject->update_key(stdString2zs(subinput->get_uuid_path()));
                bool ret = subinput->set_output("port", std::move(spObject));
                assert(ret);
                ret = subinput->set_output("hasValue", std::make_unique<NumericObject>(true));
                assert(ret);
            }
        }
        else {
            //primitive type
            auto iter2 = m_inputPrims.find(subinput_node);
            if (iter2 != m_inputPrims.end()) {
                bool ret = subinput->set_primitive_output("port", iter2->second.result);
                assert(ret);
                ret = subinput->set_output("hasValue", std::make_unique<NumericObject>(true));
                assert(ret);
            }
            else {
                subinput->set_output("port", std::make_unique<DummyObject>());
                subinput->set_output("hasValue", std::make_unique<NumericObject>(false));
            }
        }
    }

    std::set<std::string> nodesToExec;
    for (auto const &suboutput_node: m_subgraph->getSubOutputs()) {
        nodesToExec.insert(suboutput_node);
    }

    //子图的list/dict更新如何处理？
    zeno::render_reload_info _;
    m_subgraph->applyNodes(nodesToExec, _);

    //TODO: 多输出其实是一个问题，不知道view哪一个，所以目前先规定子图只能有一个输出
    auto suboutputs = m_subgraph->getSubOutputs();
    bool bSetOutput = false;
    for (auto const &suboutput_node: suboutputs) {
        auto suboutput = m_subgraph->getNode(suboutput_node);
        //suboutput的结果是放在Input的port上面（因为Suboutput放一个输出参数感觉怪怪的）
        bool bPrimoutput = suboutput->get_input_object_params().empty();
        zany result = suboutput->clone_input("port");
        if (result) {
            bSetOutput = true;
            zany spObject = result->clone();
            if (!bPrimoutput) {
                spObject->update_key(stdString2zs(get_uuid_path()));
            }
            bool ret = set_output(suboutput_node, std::move(spObject));
            assert(ret);
        }
    }

    if (m_bClearSubnet) {
        //所有子图的节点都移除对象并标脏
        m_subgraph->markDirtyAndCleanup();
    }
}

void SubnetNode::cleanInternalCaches() {
    //所有子图的节点都移除对象并标脏
    m_subgraph->markDirtyAndCleanup();
}

NodeData SubnetNode::exportInfo() const {
    //要注意，这里必须要手动cast为SubnetNode才能拿，因为NodeImpl已经和INode分离了
    NodeData node = NodeImpl::exportInfo();
    node.bclearsbn = m_bClearSubnet;
    const Asset& asset = zeno::getSession().assets->getAsset(node.cls);
    if (!asset.m_info.name.empty()) {
        node.asset = asset.m_info;
        if (in_asset_file()) {
            node.type = Node_AssetReference;
            node.bLocked = true;    //资产图里的资产只是引用，故不能展开，自然不能解锁
        }
        else {
            node.type = Node_AssetInstance;
            node.bLocked = m_bLocked;
            if (!m_bLocked) {
                node.subgraph = m_subgraph->exportGraph();
            }
        }
    }
    else {
        node.subgraph = m_subgraph->exportGraph();
        node.type = Node_SubgraphNode;
    }
    //node.customUi = m_customUi;
    return node;
}

CustomUI SubnetNode::get_customui() const
{
    return m_customUi;
}

CustomUI SubnetNode::export_customui() const {
    CustomUI exportCustomui = m_customUi;
    if (m_subgraph) {
        for (auto& tab : exportCustomui.inputPrims) {
            for (auto& group : tab.groups) {
                for (auto& param : group.params) {
                    param = get_input_prim_param(param.name);
                }
            }
        }
        for (auto& param : exportCustomui.inputObjs) {
            param = get_input_obj_param(param.name);
        }
        for (auto& param : exportCustomui.outputPrims) {
            param = get_output_prim_param(param.name);
        }
        for (auto& param : exportCustomui.outputObjs) {
            param = get_output_obj_param(param.name);
        }
    }
    return exportCustomui;
}

void SubnetNode::setCustomUi(const CustomUI& ui)
{
    m_customUi = ui;
    //保证颜色图标
    m_customUi.uistyle.background = "#1D5F51";
    m_customUi.uistyle.iconResPath = ":/icons/node/subnet.svg";
}

}
