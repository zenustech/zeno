#include <zeno/core/INode.h>
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


namespace zeno {

ZENO_API SubnetNode::SubnetNode() : subgraph(std::make_shared<Graph>(""))
{
    subgraph->optParentSubgNode = this;

    auto cl = safe_at(getSession().nodeClasses, "Subnet", "node class name").get();
    m_customUi = cl->m_customui;
}

ZENO_API SubnetNode::~SubnetNode() = default;

ZENO_API void SubnetNode::initParams(const NodeData& dat)
{
    INode::initParams(dat);
    //需要检查SubInput/SubOutput是否对的上？
    if (dat.subgraph && subgraph->getNodes().empty())
        subgraph->init(*dat.subgraph);
}

ZENO_API std::shared_ptr<Graph> SubnetNode::get_graph() const
{
    return subgraph;
}

ZENO_API bool SubnetNode::isAssetsNode() const {
    return subgraph->isAssets();
}

ZENO_API params_change_info SubnetNode::update_editparams(const ParamsUpdateInfo& params)
{
    params_change_info changes = INode::update_editparams(params);
    //update subnetnode.
    for (auto name : changes.new_inputs) {
        if (!subgraph->getNode(name)) {
            std::shared_ptr<INode> newNode = subgraph->createNode("SubInput", name);

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
                paramObj.type = Obj_Wildcard;
                paramObj.socketType = zeno::Socket_WildCard;
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
    }
    for (const auto& [old_name, new_name] : changes.rename_inputs) {
        subgraph->updateNodeName(old_name, new_name);
    }
    for (auto name : changes.remove_inputs) {
        subgraph->removeNode(name);
    }

    for (auto name : changes.new_outputs) {
        if (!subgraph->getNode(name)) {
            std::shared_ptr<INode> newNode = subgraph->createNode("SubOutput", name);

            bool exist;
            bool isprim = isPrimitiveType(false, name, exist);
            if (isprim) {
                zeno::ParamPrimitive primitive;
                primitive.bInput = true;
                primitive.name = "port";
                primitive.type = Param_Wildcard;
                primitive.socketType = Socket_WildCard;
                newNode->add_input_prim_param(primitive);
            }
            else if (!isprim && exist) {
                zeno::ParamObject paramObj;
                paramObj.bInput = true;
                paramObj.name = "port";
                paramObj.type = Obj_Wildcard;
                paramObj.socketType = zeno::Socket_WildCard;
                newNode->add_input_obj_param(paramObj);
            }
            params_change_info changes;
            changes.new_inputs.insert("port");
            changes.inputs.push_back("port");
            newNode->update_layout(changes);
        }
    }
    for (const auto& [old_name, new_name] : changes.rename_outputs) {
        subgraph->updateNodeName(old_name, new_name);
    }
    for (auto name : changes.remove_outputs) {
        subgraph->removeNode(name);
    }
    return changes;
}

void SubnetNode::mark_subnetdirty(bool bOn)
{
    if (bOn) {
        subgraph->markDirtyAll();
    }
}

ZENO_API void SubnetNode::apply() {
    for (auto const &subinput_node: subgraph->getSubInputs()) {
        auto subinput = subgraph->getNode(subinput_node);
        auto iter = m_inputObjs.find(subinput_node);
        if (iter != m_inputObjs.end()) {
            //object type.
            zany spObject = iter->second.spObject;
            bool ret = subinput->set_output("port", spObject);
            assert(ret);
            ret = subinput->set_output("hasValue", std::make_shared<NumericObject>(true));
            assert(ret);
        }
        else {
            //primitive type
            auto iter2 = m_inputPrims.find(subinput_node);
            if (iter2 != m_inputPrims.end()) {
                bool ret = subinput->set_primitive_output("port", iter2->second.result);
                assert(ret);
                ret = subinput->set_output("hasValue", std::make_shared<NumericObject>(true));
                assert(ret);
            }
            else {
                subinput->set_output("port", std::make_shared<DummyObject>());
                subinput->set_output("hasValue", std::make_shared<NumericObject>(false));
            }
        }
    }

    std::set<std::string> nodesToExec;
    for (auto const &suboutput_node: subgraph->getSubOutputs()) {
        nodesToExec.insert(suboutput_node);
    }
    subgraph->applyNodes(nodesToExec);

    for (auto const &suboutput_node: subgraph->getSubOutputs()) {
        auto suboutput = subgraph->getNode(suboutput_node);
        zany result = suboutput->get_input("port");
        if (result) {
            bool ret = set_output(suboutput_node, result);
            assert(ret);
        }
    }
}

ZENO_API NodeData SubnetNode::exportInfo() const {
    NodeData node = INode::exportInfo();
    Asset asset = zeno::getSession().assets->getAsset(node.cls);
    if (!asset.m_info.name.empty()) {
        node.asset = asset.m_info;
        node.type = Node_AssetInstance;
    }
    else {
        node.subgraph = subgraph->exportGraph();
        node.type = Node_SubgraphNode;
    }
    //node.customUi = m_customUi;
    return node;
}

ZENO_API CustomUI SubnetNode::get_customui() const
{
    return m_customUi;
}

ZENO_API CustomUI SubnetNode::export_customui() const {
    return m_customUi;
}

ZENO_API void SubnetNode::setCustomUi(const CustomUI& ui)
{
    m_customUi = ui;
}


DopNetwork::DopNetwork() : m_bEnableCache(true), m_bAllowCacheToDisk(false), m_maxCacheMemoryMB(5000), m_currCacheMemoryMB(5000), m_totalCacheSizeByte(0)
{
}

ZENO_API void DopNetwork::apply()
{
    auto& sess = zeno::getSession();
    int startFrame = sess.globalState->getStartFrame();
    int currentFarme = sess.globalState->getFrameId();
    zeno::scope_exit sp([&currentFarme, &sess]() {
        sess.globalState->updateFrameId(currentFarme);
    });
    //重新计算
    for (int i = startFrame; i <= currentFarme; i++) {
        if (m_frameCaches.find(i) == m_frameCaches.end()) {
            sess.globalState->updateFrameId(i);
            subgraph->markDirtyAll();
            zeno::SubnetNode::apply();

            const ObjectParams& outputObjs = get_output_object_params();
            size_t currentFrameCacheSize = 0;
            for (auto const& objparam : outputObjs) {
                currentFrameCacheSize += getObjSize(get_output_obj(objparam.name));
            }
            while (((m_totalCacheSizeByte + currentFrameCacheSize) / 1024 / 1024) > m_currCacheMemoryMB) {
                if (!m_frameCaches.empty()) {
                    auto lastIter = --m_frameCaches.end();
                    if (lastIter->first > currentFarme) {//先从最后一帧删

                        m_totalCacheSizeByte = m_totalCacheSizeByte - m_frameCacheSizes[lastIter->first];
                        CALLBACK_NOTIFY(dopnetworkFrameRemoved, lastIter->first)
                        m_frameCaches.erase(lastIter);
                        m_frameCacheSizes.erase(--m_frameCacheSizes.end());
                    } else {
                        m_totalCacheSizeByte = m_totalCacheSizeByte - m_frameCacheSizes.begin()->second;
                        CALLBACK_NOTIFY(dopnetworkFrameRemoved, m_frameCaches.begin()->first)
                        m_frameCaches.erase(m_frameCaches.begin());
                        m_frameCacheSizes.erase(m_frameCacheSizes.begin());
                    }
                }
                else {
                    break;
                }
            }
            for (auto const& objparam : outputObjs) {
                m_frameCaches[i].insert({ objparam.name, get_output_obj(objparam.name) });
            }
            m_frameCacheSizes[i] = currentFrameCacheSize;
            m_totalCacheSizeByte += currentFrameCacheSize;
            CALLBACK_NOTIFY(dopnetworkFrameCached, i)
        }
        else {
            if (i == currentFarme) {
                for (auto const& [name, obj] : m_frameCaches[i]) {
                    if (obj) {
                        bool ret = set_output(name, obj);
                        assert(ret);
                    }
                }
            }
        }
    }
}

ZENO_API void DopNetwork::setEnableCache(bool enable)
{
    m_bEnableCache = enable;
}

ZENO_API void DopNetwork::setAllowCacheToDisk(bool enable)
{
    m_bAllowCacheToDisk = enable;
}

ZENO_API void DopNetwork::setMaxCacheMemoryMB(int size)
{
    m_maxCacheMemoryMB = size;
}

ZENO_API void DopNetwork::setCurrCacheMemoryMB(int size)
{
    m_currCacheMemoryMB = size;
}

template <class T0>
size_t getAttrVectorSize(zeno::AttrVector<T0> const& arr) {
    size_t totalSize = 0;
    totalSize += sizeof(arr);
    totalSize += sizeof(T0) * arr.values.size();
    for (const auto& pair : arr.attrs) {
        totalSize += sizeof(pair.first) + pair.first.capacity();
        std::visit([&totalSize](auto& val) {
            if (!val.empty()) {
                using T = std::decay_t<decltype(val[0])>;
                totalSize += sizeof(T) * val.size();
            }
        }, pair.second);
    }
    return totalSize;
};

size_t DopNetwork::getObjSize(std::shared_ptr<IObject> obj)
{
    size_t totalSize = 0;
    if (std::shared_ptr<PrimitiveObject> spPrimObj = std::dynamic_pointer_cast<PrimitiveObject>(obj)) {
        PrimitiveObject* primobj = spPrimObj.get();
        totalSize += sizeof(*primobj);
        totalSize += getAttrVectorSize(primobj->verts);
        totalSize += getAttrVectorSize(primobj->points);
        totalSize += getAttrVectorSize(primobj->lines);
        totalSize += getAttrVectorSize(primobj->tris);
        totalSize += getAttrVectorSize(primobj->quads);
        totalSize += getAttrVectorSize(primobj->loops);
        totalSize += getAttrVectorSize(primobj->polys);
        totalSize += getAttrVectorSize(primobj->edges);
        totalSize += getAttrVectorSize(primobj->uvs);
        if (MaterialObject* mtlPtr = primobj->mtl.get()) {
            totalSize += sizeof(*mtlPtr);
            totalSize += mtlPtr->serializeSize();
        }
        if (InstancingObject* instPtr = primobj->inst.get()) {
            totalSize += sizeof(*instPtr);
            totalSize += instPtr->serializeSize();
        }
    }
    else if (std::shared_ptr<NumericObject> spobj = std::dynamic_pointer_cast<NumericObject>(obj)) {
        NumericObject* obj = spobj.get();
        totalSize += sizeof(*obj);
        std::visit([&totalSize](auto const& val) {
            using T = std::decay_t<decltype(val)>;
            totalSize += sizeof(val);
        }, obj->value);
    }
    else if (std::shared_ptr<StringObject> spobj = std::dynamic_pointer_cast<StringObject>(obj)) {
        StringObject* obj = spobj.get();
        totalSize += sizeof(*obj);
        totalSize += sizeof(obj->value.size());
    }
    else if (std::shared_ptr<CameraObject> spobj = std::dynamic_pointer_cast<CameraObject>(obj)) {
        CameraObject* obj = spobj.get();
        totalSize += sizeof(*obj);
        totalSize += sizeof(CameraData);
    }
    else if (std::shared_ptr<LightObject> spobj = std::dynamic_pointer_cast<LightObject>(obj)) {
        LightObject* obj = spobj.get();
        totalSize += sizeof(*obj);
        totalSize += sizeof(LightData);
    }
    else if (std::shared_ptr<MaterialObject> spobj = std::dynamic_pointer_cast<MaterialObject>(obj)) {
        MaterialObject* obj = spobj.get();
        totalSize += sizeof(*obj);
        totalSize += obj->serializeSize();
    }
    else if (std::shared_ptr<ListObject> spobj = std::dynamic_pointer_cast<ListObject>(obj)) {
        ListObject* obj = spobj.get();
        totalSize += sizeof(*obj);
        totalSize += obj->dirtyIndiceSize() * sizeof(int);
        for (int i = 0; i > obj->size(); i++) {
            totalSize += getObjSize(obj->get(i));
        }
    }
    else {//dummy obj
    }
    return totalSize;
}

void DopNetwork::resetFrameState()
{
    for (auto& [idx, _] : m_frameCaches) {
        CALLBACK_NOTIFY(dopnetworkFrameRemoved, idx)
    }
    m_frameCaches.clear();
    m_frameCacheSizes.clear();
}

ZENDEFNODE(DopNetwork, {
    {},
    {},
    {},
    {"dop"},
});

}
