#include "GraphModel.h"
#include "uicommon.h"
#include "zassert.h"
#include "variantptr.h"
#include <zeno/utils/api.h>
#include <zeno/core/NodeImpl.h>
#include "model/GraphsTreeModel.h"
#include "model/graphsmanager.h"
#include "model/assetsmodel.h"
#include "panel/zenoimagepanel.h"
#include "parammodel.h"
#include "LinkModel.h"
#include "zenoapplication.h"
#include <zeno/extra/SubnetNode.h>
#include <zeno/core/Assets.h>
#include <zeno/core/data.h>
#include <zeno/utils/helper.h>
#include "util/uihelper.h"
#include "util/jsonhelper.h"
#include "widgets/ztimeline.h"
#include "zenomainwindow.h"
#include "viewport/displaywidget.h"
#include "declmetatype.h"
#include "command.h"
#include <zeno/core/Graph.h>
#include <zeno/core/INodeClass.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/utils/inputoutput_wrapper.h>


static void triggerView(const QString& nodepath, bool bView) {
    zeno::render_reload_info info;
    info.policy = zeno::Reload_ToggleView;
    info.current_ui_graph;  //由于这是在ui下直接点击view，因此一般都是当前图（api的情况暂不考虑）

    auto spNode = zeno::getSession().getNodeByUuidPath(nodepath.toStdString());
    assert(spNode);

    zeno::render_update_info update;
    update.reason = bView ? zeno::Update_View : zeno::Update_Remove;
    update.uuidpath_node_objkey = nodepath.toStdString();
    if (spNode) {
        auto pObject = spNode->get_default_output_object();
        if (pObject) {
            update.spObject = pObject->clone();
        }
    }
    if (update.reason == zeno::Update_Remove)
    {
        //删除节点和绘制清除是异步执行，故不能用节点数据，必须要导出删除object(或者是list/dict下所有objs）的信息
        auto& sess = zeno::getSession();
        if (update.spObject) {
            update.remove_objs = zeno::get_obj_paths(update.spObject.get());
        }
    }
    info.objs.push_back(update);

    const auto& views = zenoApp->getMainWindow()->viewports();
    for (DisplayWidget* view : views) {
        view->reload(info);
    }

    const auto& images = zenoApp->getMainWindow()->imagepanels();
    for (ZenoImagePanel* panel : images) {
        panel->reload(info);
    }
}


struct NodeItem : public QObject
{
    Q_OBJECT

public:
    //temp cached data for spNode->...
    QString name;
    QString dispName;
    QString dispIcon;
    zeno::ObjPath uuidPath;
    QString cls;
    QPointF pos;

    std::string m_cbSetPos;
    std::string m_cbSetView;
    std::string m_cbSetByPass;
    std::string m_cbSetNoCache;
    std::string m_cbSetClearSbn;
    //for DopnetWork
    std::string m_cbFrameCached;
    std::string m_cbFrameRemoved;
    std::string m_cbLockChanged;

    zeno::NodeImpl* m_wpNode = nullptr;
    ParamsModel* params = nullptr;
    bool bView = false;
    bool bByPass = false;
    bool bNoCache = false;
    bool bClearSbn = false;
    bool bCollasped = false;
    bool bLoaded = true;   //如果节点所处的插件模块被卸载了，此项为false
    QmlNodeRunStatus::Value runStatus;
    zeno::NodeUIStyle uistyle;

    //for subgraph, but not include assets:
    std::optional<GraphModel*> optSubgraph;

    NodeItem(QObject* parent);
    ~NodeItem();
    void init(GraphModel* pGraphM, zeno::NodeImpl* spNode);
    void init_subgraph(GraphModel* parentGraphM);
    QString getName() {
        return name;
    }

private:
    void unregister();
};

NodeItem::NodeItem(QObject* parent) : QObject(parent)
{
}

NodeItem::~NodeItem()
{
    unregister();
}

void NodeItem::unregister()
{
    if (auto spNode = m_wpNode)
    {
        bool ret = spNode->unregister_set_pos(m_cbSetPos);
        ZASSERT_EXIT(ret);
        ret = spNode->unregister_set_view(m_cbSetView);
        ZASSERT_EXIT(ret);
        ret = spNode->unregister_set_bypass(m_cbSetByPass);
        ZASSERT_EXIT(ret);
        ret = spNode->unregister_set_nocache(m_cbSetNoCache);
        ZASSERT_EXIT(ret);

        zeno::NodeType nodetype = spNode->nodeType();
        if (nodetype == zeno::Node_SubgraphNode ||
            nodetype == zeno::Node_AssetReference ||
            nodetype == zeno::Node_AssetInstance) {
            auto subnetnode = static_cast<zeno::SubnetNode*>(spNode);
            bool ret = subnetnode->unregister_clearSubnetChanged(m_cbSetClearSbn);
            ZASSERT_EXIT(ret);
        }

        //DopNetwork
        /*
        if (auto subnetnode = dynamic_cast<zeno::DopNetwork*>(spNode))
        {
            bool ret = subnetnode->unregister_dopnetworkFrameRemoved(m_cbFrameRemoved);
            ZASSERT_EXIT(ret);
            ret = subnetnode->unregister_dopnetworkFrameCached(m_cbFrameCached);
            ZASSERT_EXIT(ret);
        }
        */
    }
    m_cbSetPos = "";
    m_cbSetView = "";
}

void NodeItem::init(GraphModel* pGraphM, zeno::NodeImpl* spNode)
{
    this->m_wpNode = spNode;

    m_cbSetPos = spNode->register_set_pos([=](std::pair<float, float> pos) {
        this->pos = { pos.first, pos.second };  //update the cache
        QModelIndex idx = pGraphM->indexFromName(this->name);
        emit pGraphM->dataChanged(idx, idx, QVector<int>{ QtRole::ROLE_OBJPOS });
    });

    m_cbSetView = spNode->register_set_view([=](bool bView) {
        this->bView = bView;
        QModelIndex idx = pGraphM->indexFromName(this->name);
        emit pGraphM->dataChanged(idx, idx, QVector<int>{ QtRole::ROLE_NODE_ISVIEW });

        //直接就在这里触发绘制更新，不再往外传递了
        //不过有一种例外，就是当前打view会触发计算的话，后续由计算触发绘制更新就可以了
        auto& session = zeno::getSession();
        if (bView && session.is_auto_run() && idx.data(QtRole::ROLE_NODE_DIRTY).toBool()) {
            return;
        }
        const QString& nodepath = idx.data(QtRole::ROLE_NODE_UUID_PATH).toString();
        triggerView(nodepath, bView);
    });

    m_cbSetByPass = spNode->register_set_bypass([=](bool bypass) {
        this->bByPass = bypass;
        QModelIndex idx = pGraphM->indexFromName(this->name);
        emit pGraphM->dataChanged(idx, idx, QVector<int>{ QtRole::ROLE_NODE_BYPASS });
    });

    m_cbSetNoCache = spNode->register_set_nocache([=](bool nocache) {
        this->bNoCache = nocache;
        QModelIndex idx = pGraphM->indexFromName(this->name);
        emit pGraphM->dataChanged(idx, idx, QVector<int>{ QtRole::ROLE_NODE_NOCACHE });
        });

    spNode->register_update_load_info([=](bool bDisable) {
        this->bLoaded = !bDisable;
        QModelIndex idx = pGraphM->indexFromName(this->name);
        emit pGraphM->dataChanged(idx, idx, QVector<int>{ QtRole::ROLE_NODE_IS_LOADED });
    });

    this->params = new ParamsModel(spNode, this);
    this->name = QString::fromStdString(spNode->get_name());
    this->cls = QString::fromStdString(spNode->get_nodecls());
    this->dispName = QString::fromStdString(spNode->get_show_name());
    this->dispIcon = QString::fromStdString(spNode->get_show_icon());
    this->bView = spNode->is_view();
    this->bByPass = spNode->is_bypass();
    this->bNoCache = spNode->is_nocache();
    this->runStatus = static_cast<QmlNodeRunStatus::Value>(spNode->get_run_status());
    auto pair = spNode->get_pos();
    this->pos = QPointF(pair.first, pair.second);
    this->uuidPath = spNode->get_uuid_path();
    this->uistyle = spNode->export_customui().uistyle;
    this->bLoaded = spNode->is_loaded();

    setProperty("uuid-path", QString::fromStdString(uuidPath));
    init_subgraph(pGraphM);

    zeno::NodeType nodetype = spNode->nodeType();
    if (nodetype == zeno::Node_SubgraphNode ||
        nodetype == zeno::Node_AssetReference ||
        nodetype == zeno::Node_AssetInstance) {
        auto subnetnode = static_cast<zeno::SubnetNode*>(spNode);
        this->bClearSbn = subnetnode->is_clearsubnet();
        m_cbSetClearSbn = subnetnode->register_clearSubnetChanged([=](bool bClearSbn) {
            this->bClearSbn = bClearSbn;
            QModelIndex idx = pGraphM->indexFromName(this->name);
            emit pGraphM->dataChanged(idx, idx, QVector<int>{ QtRole::ROLE_NODE_CLEARSUBNET });
            });
    }

    //DopNetwork
    /*
    if (auto subnetnode = dynamic_cast<zeno::DopNetwork*>(spNode))
    {
        m_cbFrameRemoved = subnetnode->register_dopnetworkFrameRemoved([](int frame) {
            ZenoMainWindow* mainWin = zenoApp->getMainWindow();
            ZASSERT_EXIT(mainWin);
            ZTimeline* timeline = mainWin->timeline();
            ZASSERT_EXIT(timeline);
            timeline->updateDopnetworkFrameRemoved(frame);
        });
        m_cbFrameCached = subnetnode->register_dopnetworkFrameCached([](int frame) {
            ZenoMainWindow* mainWin = zenoApp->getMainWindow();
            ZASSERT_EXIT(mainWin);
            ZTimeline* timeline = mainWin->timeline();
            ZASSERT_EXIT(timeline);
            timeline->updateDopnetworkFrameCached(frame);
        });
    }
    */
}

void NodeItem::init_subgraph(GraphModel* parentGraphM) {
    if (auto subnetnode = dynamic_cast<zeno::SubnetNode*>(m_wpNode))
    {
        std::string const& subnetpath = m_wpNode->get_path();
        auto pModel = new GraphModel(subnetpath, false, parentGraphM->treeModel(), parentGraphM, this);
        bool bAssets = subnetnode->get_subgraph()->isAssets();
        if (bAssets) {
            m_cbLockChanged = subnetnode->register_lockChanged([=]() {
                QModelIndex idx = parentGraphM->indexFromName(this->name);
                bool bLocked = m_wpNode->is_locked();
                emit pModel->lockStatusChanged();
                emit parentGraphM->dataChanged(idx, idx, QVector<int>{ QtRole::ROLE_NODE_LOCKED });
                });
            //TODO: init value from io.
            emit pModel->lockStatusChanged();
        }
        this->optSubgraph = pModel;
    }
}

#include "GraphModel.moc"

struct GraphMImpl : QObject
{
    GraphMImpl(zeno::Graph* spGraph, GraphModel* parent)
        : QObject(parent)
        , m_wpCoreGraph(spGraph)
        , m_parentM(parent)
    {
    }

    zeno::Graph* m_wpCoreGraph;
    GraphModel* m_parentM;
};

GraphModel::GraphModel(std::string const& asset_or_graphpath, bool bAsset, GraphsTreeModel* pTree, GraphModel* parentGraph, QObject* parent)
    : QAbstractListModel(parent)
    , m_pTree(pTree)
{
    std::shared_ptr<zeno::Graph> spGraph;
    if (bAsset) {
        auto& assets = zeno::getSession().assets;
        spGraph = assets->getAssetGraph(asset_or_graphpath, true);
    }
    else {
        spGraph = zeno::getSession().getGraphByPath(asset_or_graphpath);
    }

    m_impl.reset(new GraphMImpl(spGraph.get(), parentGraph));

    m_graphName = QString::fromStdString(spGraph->getName());
    m_undoRedoStack = m_graphName == "main" || zeno::getSession().assets->isAssetGraph(spGraph.get()) ? new QUndoStack(this) : nullptr;
    m_linkModel = new LinkModel(this);
    registerCoreNotify();

    std::map<std::string, zeno::NodeImpl*> nodes;
    nodes = spGraph->getNodes();

    for (auto& [name, node] : nodes) {
        _appendNode(node);
    }

    _initLink();

    if (m_pTree) {
        connect(this, SIGNAL(rowsInserted(const QModelIndex&, int, int)),
            m_pTree, SLOT(onGraphRowsInserted(const QModelIndex&, int, int)));
        connect(this, SIGNAL(rowsAboutToBeRemoved(const QModelIndex&, int, int)),
            m_pTree, SLOT(onGraphRowsAboutToBeRemoved(const QModelIndex&, int, int)));
        connect(this, SIGNAL(rowsRemoved(const QModelIndex&, int, int)),
            m_pTree, SLOT(onGraphRowsRemoved(const QModelIndex&, int, int)));
        connect(this, SIGNAL(nameUpdated(const QModelIndex&, const QString&)),
            m_pTree, SLOT(onNameUpdated(const QModelIndex&, const QString&)));
    }
}

void GraphModel::registerCoreNotify()
{
    auto coreGraph = m_impl->m_wpCoreGraph;
    ZASSERT_EXIT(coreGraph);

    m_cbCreateNode = coreGraph->register_createNode([&](const std::string& name, zeno::NodeImpl* spNode) {
        _appendNode(spNode);
    });

    m_cbRemoveNode = coreGraph->register_removeNode([&](const std::string& name) {
        //before core remove
        QString qName = QString::fromStdString(name);
        ZASSERT_EXIT(m_name2uuid.find(qName) != m_name2uuid.end(), false);
        QString uuid = m_name2uuid[qName];
        ZASSERT_EXIT(m_uuid2Row.find(uuid) != m_uuid2Row.end(), false);
        int row = m_uuid2Row[uuid];
        NodeItem* pItem = m_nodes[uuid];
        if (pItem && pItem->bView) {
            const QString& nodepath = QString::fromStdString(pItem->uuidPath);
            triggerView(nodepath, false);
        }
        removeRow(row);
        zenoApp->graphsManager()->currentModel()->markDirty(true);
    });

    m_cbAddLink = coreGraph->register_addLink([&](zeno::EdgeInfo edge) -> bool {
        _addLink_callback(edge);
        return true;
    });

    m_cbRenameNode = coreGraph->register_updateNodeName([&](std::string oldname, std::string newname) {
        const QString& oldName = QString::fromStdString(oldname);
        const QString& newName = QString::fromStdString(newname);
        _updateName(oldName, newName);
    });

    m_cbRemoveLink = coreGraph->register_removeLink([&](zeno::EdgeInfo edge) -> bool {
        return _removeLink(edge);
    });

    m_cbClearGraph = coreGraph->register_clear([&]() {
        //begin clear.
        _clear();
    });
}

void GraphModel::unRegisterCoreNotify()
{
    if (auto coreGraph = m_impl->m_wpCoreGraph)
    {
        bool ret = coreGraph->unregister_createNode(m_cbCreateNode);
        ZASSERT_EXIT(ret);
        ret = coreGraph->unregister_removeNode(m_cbRemoveNode);
        ZASSERT_EXIT(ret);
        ret = coreGraph->unregister_addLink(m_cbAddLink);
        ZASSERT_EXIT(ret);
        ret = coreGraph->unregister_removeLink(m_cbRemoveLink);
        ZASSERT_EXIT(ret);
        ret = coreGraph->unregister_updateNodeName(m_cbRenameNode);
        ZASSERT_EXIT(ret);
        ret = coreGraph->unregister_clear(m_cbClearGraph);
        ZASSERT_EXIT(ret);
    }
    m_cbCreateNode = "";
    m_cbCreateNode = "";
    m_cbAddLink = "";
    m_cbRemoveLink = "";
    m_cbRenameNode = "";
    m_cbClearGraph = "";
}

GraphModel::~GraphModel()
{
    unRegisterCoreNotify();
}

void GraphModel::clear()
{
    auto spGraph = m_impl->m_wpCoreGraph;
    ZASSERT_EXIT(spGraph);
    spGraph->clear();
}

QmlParamType::Value GraphModel::getParamType(const QString& node, bool bInput, const QString& param) const {
    if (m_name2uuid.find(node) == m_name2uuid.end()) {
        zeno::log_error("error type");
        return QmlParamType::Unknown;
    }

    QString uuid = m_name2uuid[node];
    NodeItem* item = m_nodes[uuid];
    QModelIndex paramIdx = item->params->paramIdx(param, bInput);
    if (!paramIdx.isValid()) {
        return QmlParamType::Unknown;
    }
    zeno::ParamType type = paramIdx.data(QtRole::ROLE_PARAM_TYPE).toLongLong();
    switch (type)
    {
    case zeno::types::gParamType_String:    return QmlParamType::String;
    case zeno::types::gParamType_Int:       return QmlParamType::Int;
    case zeno::types::gParamType_Float:     return QmlParamType::Float;
    case zeno::types::gParamType_Vec3f:     return QmlParamType::Vec3f;
    case gParamType_List:       return QmlParamType::List;
    case gParamType_Dict:       return QmlParamType::Dict;
    case gParamType_Geometry:   return QmlParamType::Geometry;
    case gParamType_IObject:    return QmlParamType::IObject;
    default:
        return QmlParamType::Unknown;
    }
}

int GraphModel::indexFromId(const QString& name) const
{
    if (m_name2uuid.find(name) == m_name2uuid.end())
        return -1;

    QString uuid = m_name2uuid[name];
    if (m_uuid2Row.find(uuid) == m_uuid2Row.end())
        return -1;
    return m_uuid2Row[uuid];
}

QModelIndex GraphModel::indexFromName(const QString& name) const {
    int row = indexFromId(name);
    if (row == -1) {
        return QModelIndex();
    }
    return createIndex(row, 0);
}

QModelIndex GraphModel::indexFromUuid(const QString& uuid) const {
    auto iter = m_uuid2Row.find(uuid);
    if (iter == m_uuid2Row.end()) return QModelIndex();
    return createIndex(iter.value(), 0);
}

QModelIndex GraphModel::indexFromUuidPath(const zeno::ObjPath& uuidPath) const {
    //uuidpath是这种：xxxx-subnet/xxx-attributewrangle
    if (uuidPath.empty())
        return QModelIndex();

    int idx = uuidPath.find("/");
    const QString& uuid = QString::fromStdString(uuidPath.substr(0, idx));
    if (m_nodes.find(uuid) != m_nodes.end()) {
        NodeItem* pItem = m_nodes[uuid];
        zeno::ObjPath _path = uuidPath;
        if (idx < 0) {
            return createIndex(m_uuid2Row[uuid], 0, nullptr);
        }
        else if (pItem->optSubgraph.has_value()) {
            _path = uuidPath.substr(idx + 1, uuidPath.size() - idx);
            return pItem->optSubgraph.value()->indexFromUuidPath(_path);
        }
    }
    return QModelIndex();
}

void GraphModel::indiceFromUuidPath(const zeno::ObjPath& uuidPath, QModelIndexList& pathNodes) const {
    if (uuidPath.empty())
        return;

    int idx = uuidPath.find("/");
    const QString& uuid = QString::fromStdString(uuidPath.substr(0, idx));
    if (m_nodes.find(uuid) != m_nodes.end()) {
        NodeItem* pItem = m_nodes[uuid];
        zeno::ObjPath _path = uuidPath;
        if (idx < 0) {
            QModelIndex finalNode = createIndex(m_uuid2Row[uuid], 0, nullptr);
            pathNodes.append(finalNode);
        }
        else if (pItem->optSubgraph.has_value()) {
            _path = uuidPath.substr(idx + 1, uuidPath.size() - idx);
            QModelIndex subnetNode = createIndex(m_uuid2Row[uuid], 0, nullptr);
            pathNodes.append(subnetNode);
            pItem->optSubgraph.value()->indiceFromUuidPath(_path, pathNodes);
        }
    }
}

std::set<std::string> GraphModel::getViewNodePath() const {
    auto viewnodenames = m_impl->m_wpCoreGraph->get_viewnodes();
    std::set<std::string> nodes;
    for (auto name : viewnodenames) {
        QModelIndex idx = indexFromName(QString::fromStdString(name));
        const QString& nodepath = idx.data(QtRole::ROLE_NODE_UUID_PATH).toString();
        nodes.insert(nodepath.toStdString());
    }
    return nodes;
}

void GraphModel::addLink(
    const QString& outNode,
    const QString& outParam,
    const QString& inNode,
    const QString& inParam,
    bool asListElement)
{
    if (outNode == inNode)
        return;

    zeno::EdgeInfo link;
    link.inNode = inNode.toStdString();
    link.inParam = inParam.toStdString();
    link.outNode = outNode.toStdString();
    link.outParam = outParam.toStdString();

    QmlParamType::Value outparam_type = getParamType(outNode, false, outParam);
    QmlParamType::Value inparam_type = getParamType(inNode, true, inParam);
    if (QmlParamType::List == inparam_type) {
        QModelIndex inNodeIdx = indexFromName(inNode);
        QString uuid = m_name2uuid[inNode];
        QModelIndex inParamIdx = m_nodes[uuid]->params->paramIdx(inParam, true);
        PARAM_LINKS links = inParamIdx.data(QtRole::ROLE_LINKS).value<PARAM_LINKS>();
        if ((outparam_type == QmlParamType::List || outparam_type == QmlParamType::IObject) && !asListElement) {
            link.inKey = "";
        }
        else {
            link.inKey = "obj0";    //内核代码会自动纠正
        }
    }
    _addLink_apicall(link, true);
}

void GraphModel::addLink(const zeno::EdgeInfo& link)
{
    _addLink_apicall(link, true);
}

QString GraphModel::name() const
{
    return m_graphName;
}

QStringList GraphModel::path() const
{
    return currentPath();
}

void GraphModel::insertNode(const QString& nodeCls, const QString& cate, const QPointF& pos) {
    zeno::NodeData dat = createNode(nodeCls, cate, pos);
}

QString GraphModel::owner() const
{
    if (auto pItem = qobject_cast<NodeItem*>(parent()))
    {
        auto spNode = pItem->m_wpNode;
        return spNode ? QString::fromStdString(spNode->get_name()) : "";
    }
    else {
        return "main";
    }
}

int GraphModel::rowCount(const QModelIndex& parent) const
{
    return m_nodes.size();
}

QVariant GraphModel::data(const QModelIndex& index, int role) const
{
    NodeItem* item = m_nodes[m_row2uuid[index.row()]];

    switch (role) {
        case Qt::DisplayRole:
        case QtRole::ROLE_NODE_NAME: {
            return item->name;
        }
        case QtRole::ROLE_NODE_DISPLAY_NAME: {
            return item->dispName;
        }
        case QtRole::ROLE_NODE_DISPLAY_ICON: {
            return item->dispIcon;
        }
        case QtRole::ROLE_NODE_UISTYLE: {
            QVariantMap map;
            map["icon"] = QString::fromStdString(item->uistyle.iconResPath);
            map["background"] = QString::fromStdString(item->uistyle.background);
            return map;
        }
        case QtRole::ROLE_NODE_UUID_PATH: {
            return QString::fromStdString(item->uuidPath);
        }
        case QtRole::ROLE_CLASS_NAME: {
            return item->cls;
        }
        case QtRole::ROLE_OBJPOS: {
            return item->pos;
        }
        case QtRole::ROLE_PARAMS:
        {
            return QVariantPtr<ParamsModel>::asVariant(item->params);
        }
        case QtRole::ROLE_SUBGRAPH:
        {
            if (item->optSubgraph.has_value())
                return QVariant::fromValue(item->optSubgraph.value());
            else
                return QVariant();  
        }
        case QtRole::ROLE_GRAPH:
        {
            return QVariantPtr<GraphModel>::asVariant(const_cast<GraphModel*>(this));
        }
        case QtRole::ROLE_INPUTS:
        {
            if (item->params)
                return QVariant::fromValue(item->params->getInputs());
            return QVariant();
        }
        case QtRole::ROLE_OUTPUTS:
        {
            if (item->params)
                return QVariant::fromValue(item->params->getOutputs());
            return QVariant();
        }
        case QtRole::ROLE_NODEDATA:
        {
            zeno::NodeData data;
            auto spNode = item->m_wpNode;
            ZASSERT_EXIT(spNode, QVariant());
            return QVariant::fromValue(spNode->exportInfo());
        }
        case QtRole::ROLE_NODE_LOCKED:
        {
            return item->m_wpNode->is_locked();
        }
        case QtRole::ROLE_NODE_IS_LOADED:
        {
            auto spNode = item->m_wpNode;
            return item->bLoaded;
        }
        case QtRole::ROLE_NODE_STATUS:
        {
            int options = zeno::None;
            if (item->bView)
                options |= zeno::View;
            if (item->bByPass)
                options |= zeno::ByPass;
            if (item->bClearSbn)
                options |= zeno::ClearSbn;
            if (item->bNoCache)
                options |= zeno::Nocache;
            return QVariant(options);
        }
        case QtRole::ROLE_OUTPUT_OBJS:
        {
            auto spNode = item->m_wpNode;
            ZASSERT_EXIT(spNode, QVariant());
            auto spOutObj = spNode->get_default_output_object();
            //如果有多个，只取第一个。
            if (spOutObj) {
                return QVariant::fromValue(spOutObj);
            }
            else {
                return QVariant();
            }
        }
        case QtRole::ROLE_OUTPUT_VIEWOBJ:
        {
            auto spNode = item->m_wpNode;
            ZASSERT_EXIT(spNode, QVariant());
            auto cloneObj = spNode->clone_default_output_object();
            if (cloneObj) {
                return QVariant::fromValue(cloneObj.release());
            }
            return QVariant();
        }
        case QtRole::ROLE_NODE_ISVIEW:
        {
            return item->bView;
        }
        case QtRole::ROLE_NODE_BYPASS:
        {
            return item->bByPass;
        }
        case QtRole::ROLE_NODE_NOCACHE:
        {
            return item->bNoCache;
        }
        case QtRole::ROLE_NODE_CLEARSUBNET:
        {
            return item->bClearSbn;
        }
        case QtRole::ROLE_NODE_RUN_STATE:
        {
            QmlNodeRunStatus::Value qmlstate = static_cast<QmlNodeRunStatus::Value>(item->runStatus);
            return qmlstate;
        }
        case QtRole::ROLE_NODE_DIRTY:
        {
            return item->runStatus != QmlNodeRunStatus::RunSucceed && item->runStatus != QmlNodeRunStatus::Clean;
        }
        case QtRole::ROLE_NODETYPE:
        {
            if (item->m_wpNode && item->m_wpNode->get_nodecls() == "Group")
                return zeno::Node_Group;
            zeno::NodeType type = item->m_wpNode->nodeType();
            return type;
        }
        case QtRole::ROLE_NODE_CATEGORY:
        {
            auto spNode = item->m_wpNode;
            if (spNode->nodeClass)
            {
                return QString::fromStdString(spNode->nodeClass->m_customui.category);
            }
            else
            {
                return "";
            }
        }
        case QtRole::ROLE_OBJPATH:
        {
            auto spNode = item->m_wpNode;
            if (spNode) {
                return QString::fromStdString(spNode->get_path());
            }
            //QStringList path = currentPath();
            //path.append(item->name);
            return "";
        }
        case QtRole::ROLE_COLLASPED:
        {
            return item->bCollasped;
        }
        case QtRole::ROLE_KEYFRAMES: 
        {
            QVector<int> keys;
            for (const zeno::ParamPrimitive& info : item->params->getInputs()) {
                if (!info.defl.has_value()) {
                    continue;
                }
                const QVariant& value = QVariant::fromValue(info.defl);

                bool bValid = false;
                zeno::CurvesData curves = UiHelper::getCurvesFromQVar(value, &bValid);
                if (curves.empty() || !bValid) {
                    continue;
                }

                for (auto& [_, curve] : curves.keys) {
                    QVector<int> cpbases;
                    for (auto xval : curve.cpbases) {
                        cpbases << xval;
                    }
                    keys << cpbases;
                }
            }
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            return QVariant::fromValue(keys);
        }
        //Dopnetwork
        case QtRole::ROLE_DOPNETWORK_ENABLECACHE: {
            //if (auto spNode = item->m_wpNode) {
            //    if (auto spDop = dynamic_cast<zeno::DopNetwork*>(spNode)) {
            //        return spDop->m_bEnableCache;
            //    }
            //}
            return QVariant();
        }
        case QtRole::ROLE_DOPNETWORK_CACHETODISK: {
            if (auto spNode = item->m_wpNode) {
                //if (auto spDop = dynamic_cast<zeno::DopNetwork*>(spNode)) {
                //    return spDop->m_bAllowCacheToDisk;
                //}
            }
            return QVariant();
        }
        case QtRole::ROLE_DOPNETWORK_MAXMEM: {
            //if (auto spNode = item->m_wpNode) {
            //    if (auto spDop = dynamic_cast<zeno::DopNetwork*>(spNode)) {
            //        return spDop->m_maxCacheMemoryMB;
            //    }
            //}
            return QVariant();
        }
        case QtRole::ROLE_DOPNETWORK_MEM: {
            //if (auto spNode = item->m_wpNode) {
            //    if (auto spDop = dynamic_cast<zeno::DopNetwork*>(spNode)) {
            //        return spDop->m_currCacheMemoryMB;
            //    }
            //}
            return QVariant();
        }
        default:
            return QVariant();
    }
}

bool GraphModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    NodeItem* item = m_nodes[m_row2uuid[index.row()]];

    switch (role) {
        case QtRole::ROLE_CLASS_NAME: {
            //TODO: rename by core graph
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
        case QtRole::ROLE_NODE_NAME: {
            updateNodeName(index, value.toString());
            return true;
        }
        case QtRole::ROLE_OBJPOS:
        {
            auto spNode = item->m_wpNode;
            if (value.type() == QVariant::PointF) {
                QPointF pos = value.toPoint();
                spNode->set_pos({ pos.x(), pos.y() });
            }
            else {
                QVariantList lst = value.toList();
                if (lst.size() != 2)
                    return false;
                spNode->set_pos({ lst[0].toFloat(), lst[1].toFloat() });
            }
            return true;
        }
        case QtRole::ROLE_NODE_LOCKED:
        {
            item->m_wpNode->set_locked(value.toBool());
            return true;
        }
        case QtRole::ROLE_COLLASPED:
        {
            item->bCollasped = value.toBool();
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
        case QtRole::ROLE_NODE_RUN_STATE:
        {
            item->runStatus = value.value<QmlNodeRunStatus::Value>();
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
        case QtRole::ROLE_NODE_STATUS:
        {
            setView(index, value.toInt() & zeno::View);
            // setMute();
            break;
        }
        case QtRole::ROLE_NODE_ISVIEW:
        {
            setView(index, value.toBool());
            break;
        }
        case QtRole::ROLE_NODE_BYPASS:
        {
            setBypass(index, value.toBool());
            break;
        }
        case QtRole::ROLE_NODE_NOCACHE:
        {
            setNocache(index, value.toBool());
            break;
        }
        case QtRole::ROLE_NODE_CLEARSUBNET:
        {
            setClearSubnet(index, value.toBool());
            break;
        }
        case QtRole::ROLE_INPUTS:
        {
            PARAMS_INFO paramsInfo = value.value<PARAMS_INFO>();
            for (auto&[key, param] : paramsInfo)
            {
                QModelIndex idx = item->params->paramIdx(key, true);
                item->params->setData(idx, QVariant::fromValue(param.defl), QtRole::ROLE_PARAM_VALUE);
            }
            return true;
        }
        //Dopnetwork
        case QtRole::ROLE_DOPNETWORK_ENABLECACHE: {
            //if (auto spNode = item->m_wpNode) {
            //    if (auto spDop = dynamic_cast<zeno::DopNetwork*>(spNode)) {
            //        spDop->setEnableCache(value.toBool());
            //    }
            //}
            return true;
        }
        case QtRole::ROLE_DOPNETWORK_CACHETODISK: {
            //if (auto spNode = item->m_wpNode) {
            //    if (auto spDop = dynamic_cast<zeno::DopNetwork*>(spNode)) {
            //        spDop->setAllowCacheToDisk(value.toBool());
            //    }
            //}
            return true;
        }
        case QtRole::ROLE_DOPNETWORK_MAXMEM: {
            //if (auto spNode = item->m_wpNode) {
            //    if (auto spDop = dynamic_cast<zeno::DopNetwork*>(spNode)) {
            //        spDop->setMaxCacheMemoryMB(value.toInt());
            //    }
            //}
            return true;
        }
        case QtRole::ROLE_DOPNETWORK_MEM: {
            //if (auto spNode = item->m_wpNode) {
            //    if (auto spDop = dynamic_cast<zeno::DopNetwork*>(spNode)) {
            //        spDop->setCurrCacheMemoryMB(value.toInt());
            //    }
            //}
            return true;
        }
    }
    return false;
}

QModelIndexList GraphModel::match(const QModelIndex& start, int role,
    const QVariant& value, int hits,
    Qt::MatchFlags flags) const
{
    QModelIndexList result;
    if (role == QtRole::ROLE_CLASS_NAME) {
        auto spGraph = m_impl->m_wpCoreGraph;
        ZASSERT_EXIT(spGraph, result);
        std::string content = value.toString().toStdString();
        auto results = spGraph->searchByClass(content);
        for (std::string node : results) {
            QModelIndex nodeIdx = indexFromName(QString::fromStdString(node));
            result.append(nodeIdx);
        }
    }
    return result;
}

QList<SEARCH_RESULT> GraphModel::search(const QString& content, SearchType searchType, SearchOpt searchOpts)
{
    QList<SEARCH_RESULT> results;
    if (content.isEmpty())
        return results;

    if (searchType & SEARCH_NODEID) {
        QModelIndexList lst;
        if (searchOpts == SEARCH_MATCH_EXACTLY) {
            QModelIndex idx = indexFromName(content);
            if (idx.isValid())
                lst.append(idx);
        }
        else {
            lst = _base::match(this->index(0, 0), QtRole::ROLE_NODE_NAME, content, -1, Qt::MatchContains);
        }
        if (!lst.isEmpty()) {
            for (const QModelIndex& nodeIdx : lst) {
                SEARCH_RESULT result;
                result.targetIdx = nodeIdx;
                result.subGraph = this;
                result.type = SEARCH_NODEID;
                results.append(result);
            }
        }
        for (auto& subnode : m_subgNodes)
        {
            if (m_name2uuid.find(subnode) == m_name2uuid.end())
                continue;
            NodeItem* pItem = m_nodes[m_name2uuid[subnode]];
            if (!pItem->optSubgraph.has_value())
                continue;
            const QList<SEARCH_RESULT>& subnodeRes = pItem->optSubgraph.value()->search(content, searchType, searchOpts);
            for (auto& res: subnodeRes)
                results.push_back(res);
        }
    }

    //TODO

    return results;
}

QList<SEARCH_RESULT> GraphModel::searchByUuidPath(const zeno::ObjPath& uuidPath)
{
    QList<SEARCH_RESULT> results;
    if (uuidPath.empty())
        return results;

    SEARCH_RESULT result;
    result.targetIdx = indexFromUuidPath(uuidPath);
    result.subGraph = getGraphByPath(uuidPath2ObjPath(uuidPath));
    result.type = SEARCH_NODEID;
    results.append(result);
    return results;
}

QStringList GraphModel::uuidPath2ObjPath(const zeno::ObjPath& uuidPath)
{
    QStringList res;
    zeno::ObjPath tmp = uuidPath;
    if (tmp.empty())
        return res;

    int idx = tmp.find("/");
    auto uuid = tmp.substr(0, idx);
    auto it = m_nodes.find(QString::fromStdString(uuid));
    if (it == m_nodes.end()) {
        NodeItem* pItem = it.value();
        res.append(pItem->getName());

        if (idx >= 0)
            tmp = tmp.substr(idx+1, tmp.size() - idx);

        if (pItem->optSubgraph.has_value())
            res.append(pItem->optSubgraph.value()->uuidPath2ObjPath(tmp));
    }

    return res;
}

GraphModel* GraphModel::getGraphByPath(const QStringList& objPath)
{
    //这个api非常难用，TODO: deprecated!!!
    QStringList items = objPath;
    if (items.empty())
        return this;

    QString item = items[0];

    if (m_name2uuid.find(item) == m_name2uuid.end()) {
        return nullptr;
    }

    QString uuid = m_name2uuid[item];
    auto it = m_nodes.find(uuid);
    if (it == m_nodes.end()) {
        return nullptr;
    }

    NodeItem* pItem = it.value();
    items.removeAt(0);

    if (items.isEmpty())
    {
        if (pItem->optSubgraph.has_value())
        {
            return pItem->optSubgraph.value();
        }
        else
        {
            return this;
        }
    }
    ZASSERT_EXIT(pItem->optSubgraph.has_value(), nullptr);
    return pItem->optSubgraph.value()->getGraphByPath(items);
}

QStringList GraphModel::currentPath() const
{
    if (m_graphName == "main")
        return { "main" };

    QStringList path;
    NodeItem* pNode = qobject_cast<NodeItem*>(this->parent());
    if (!pNode)
        return { this->m_graphName };

    GraphModel* pGraphM = nullptr;
    while (pNode) {
        path.push_front(pNode->name);
        pGraphM = qobject_cast<GraphModel*>(pNode->parent());
        ZASSERT_EXIT(pGraphM, {});
        pNode = qobject_cast<NodeItem*>(pGraphM->parent());
    }
    path.push_front(pGraphM->name());
    return path;
}

void GraphModel::undo()
{
    zeno::getSession().beginApiCall();
    zeno::scope_exit scope([=]() { zeno::getSession().endApiCall(); });
    if (m_undoRedoStack.has_value() && m_undoRedoStack.value())
        m_undoRedoStack.value()->undo();
}

void GraphModel::redo()
{
    zeno::getSession().beginApiCall();
    zeno::scope_exit scope([=]() { zeno::getSession().endApiCall(); });
    if (m_undoRedoStack.has_value() && m_undoRedoStack.value())
        m_undoRedoStack.value()->redo();
}

void GraphModel::pushToplevelStack(QUndoCommand* cmd)
{
    if (m_undoRedoStack.has_value() && m_undoRedoStack.value())
        m_undoRedoStack.value()->push(cmd);
}

void GraphModel::beginMacro(const QString& name)
{
    auto curpath = currentPath();
    if (curpath.size() > 1)   //不是顶层graph，则调用顶层graph
    {
        if (GraphModel* topLevelGraph = getTopLevelGraph(curpath))
            topLevelGraph->beginMacro(name);
    }
    else {
        if (m_undoRedoStack.has_value() && m_undoRedoStack.value())
            m_undoRedoStack.value()->beginMacro(name);
        zeno::getSession().beginApiCall();
    }
}

void GraphModel::endMacro()
{
    auto curpath = currentPath();
    if (curpath.size() > 1)   //不是顶层graph，则调用顶层graph
    {
        if (GraphModel* topLevelGraph = getTopLevelGraph(curpath))
            topLevelGraph->endMacro();
    }
    else {
        if (m_undoRedoStack.has_value() && m_undoRedoStack.value())
            m_undoRedoStack.value()->endMacro();
        zeno::getSession().endApiCall();
    }
}

void GraphModel::_initLink()
{
    for (auto item : m_nodes)
    {
        auto spNode = item->m_wpNode;
        ZASSERT_EXIT(spNode);
        zeno::NodeData nodedata = spNode->exportInfo();
        //objects links init
        for (auto param : nodedata.customUi.inputObjs) {
            for (auto link : param.links) {
                link.bObjLink = true;
                _addLink_callback(link);
            }
        }
        //primitives links init
        for (auto tab : nodedata.customUi.inputPrims) {
            for (auto group : tab.groups) {
                for (auto param : group.params) {
                    for (auto link : param.links) {
                        link.bObjLink = false;
                        _addLink_callback(link);
                    }
                }
            }
        }
    }
}

void GraphModel::_addLink_callback(const zeno::EdgeInfo link)
{
    QModelIndex from, to;

    QString outNode = QString::fromStdString(link.outNode);
    QString outParam = QString::fromStdString(link.outParam);
    QString outKey = QString::fromStdString(link.outKey);
    QString inNode = QString::fromStdString(link.inNode);
    QString inParam = QString::fromStdString(link.inParam);
    QString inKey = QString::fromStdString(link.inKey);

    if (m_name2uuid.find(outNode) == m_name2uuid.end() ||
        m_name2uuid.find(inNode) == m_name2uuid.end())
        return;

    ParamsModel* fromParams = m_nodes[m_name2uuid[outNode]]->params;
    ParamsModel* toParams = m_nodes[m_name2uuid[inNode]]->params;

    from = fromParams->paramIdx(outParam, false);
    to = toParams->paramIdx(inParam, true);
    
    if (from.isValid() && to.isValid())
    {
        //notify ui to create dict key slot.
        if (!link.inKey.empty())
            emit toParams->linkAboutToBeInserted(link);
        if (!link.outKey.empty()) {
            if (zenoApp->graphsManager()->isInitializing())
                emit fromParams->linkAboutToBeInserted(link);
        }

        QModelIndex linkIdx = m_linkModel->addLink(from, outKey, to, inKey, link.bObjLink);
        fromParams->addLink(from, linkIdx);
        toParams->addLink(to, linkIdx);
    }

    zenoApp->graphsManager()->currentModel()->markDirty(true);
}

bool GraphModel::removeLink(
    const QString& outNode,
    const QString& outParam,
    const QString& inNode,
    const QString& inParam,
    const QString& outKey,
    const QString& inKey)
{
    zeno::EdgeInfo edge;
    edge.inKey = inKey.toStdString();
    edge.inParam = inParam.toStdString();
    edge.inNode = inNode.toStdString();
    edge.outKey = outKey.toStdString();
    edge.outParam = outParam.toStdString();
    edge.outNode = outNode.toStdString();

    ZASSERT_EXIT(m_name2uuid.find(inNode) != m_name2uuid.end(), false);
    QString inNode_uuid = m_name2uuid[inNode];
    ZASSERT_EXIT(m_nodes.find(inNode_uuid) != m_nodes.end(), false);
    NodeItem* item = m_nodes[inNode_uuid];
    QModelIndex idxParam = item->params->index(item->params->indexFromName(inParam, true));
    edge.bObjLink = idxParam.data(QtRole::ROLE_PARAM_GROUP) == zeno::Role_InputObject;

    removeLink(edge);
    return true;
}

void GraphModel::removeLink(const QModelIndex& linkIdx)
{
    zeno::EdgeInfo edge = linkIdx.data(QtRole::ROLE_LINK_INFO).value<zeno::EdgeInfo>();
    removeLink(edge);
}

void GraphModel::removeLink(const zeno::EdgeInfo& link)
{
    _removeLinkImpl(link, true);
}

bool GraphModel::updateLink(const QModelIndex& linkIdx, bool bInput, const QString& oldkey, const QString& newkey)
{
    auto spGraph = m_impl->m_wpCoreGraph;
    ZASSERT_EXIT(spGraph, false);
    zeno::EdgeInfo edge = linkIdx.data(QtRole::ROLE_LINK_INFO).value<zeno::EdgeInfo>();
    bool ret = spGraph->updateLink(edge, bInput, oldkey.toStdString(), newkey.toStdString());
    if (!ret)
        return ret;

    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(linkIdx.model());
    LinkModel* linksM = qobject_cast<LinkModel*>(pModel);
    linksM->setData(linkIdx, newkey, QtRole::ROLE_LINK_INKEY);
}

void GraphModel::moveUpLinkKey(const QModelIndex& linkIdx, bool bInput, const std::string& keyName)
{
    auto spGraph = m_impl->m_wpCoreGraph;
    ZASSERT_EXIT(spGraph);
    zeno::EdgeInfo edge = linkIdx.data(QtRole::ROLE_LINK_INFO).value<zeno::EdgeInfo>();
    spGraph->moveUpLinkKey(edge, bInput, keyName);
}

bool GraphModel::_removeLink(const zeno::EdgeInfo& edge)
{
    QString outNode = QString::fromStdString(edge.outNode);
    QString inNode = QString::fromStdString(edge.inNode);
    QString outParam = QString::fromStdString(edge.outParam);
    QString inParam = QString::fromStdString(edge.inParam);

    ZASSERT_EXIT(m_name2uuid.find(outNode) != m_name2uuid.end(), false);
    ZASSERT_EXIT(m_name2uuid.find(inNode) != m_name2uuid.end(), false);

    QModelIndex from, to;

    ParamsModel* fromParams = m_nodes[m_name2uuid[outNode]]->params;
    ParamsModel* toParams = m_nodes[m_name2uuid[inNode]]->params;

    from = fromParams->paramIdx(outParam, false);
    to = toParams->paramIdx(inParam, true);
    if (from.isValid() && to.isValid())
    {
        emit toParams->linkAboutToBeRemoved(edge);
        QModelIndex linkIdx = fromParams->removeOneLink(from, edge);
        QModelIndex linkIdx2 = toParams->removeOneLink(to, edge);
        ZASSERT_EXIT(linkIdx == linkIdx2, false);
        if (linkIdx.isValid())
            m_linkModel->removeRow(linkIdx.row());
    }
    zenoApp->graphsManager()->currentModel()->markDirty(true);
    return true;
}

void GraphModel::_updateName(const QString& oldName, const QString& newName)
{
    ZASSERT_EXIT(oldName != newName);

    m_name2uuid[newName] = m_name2uuid[oldName];

    QString uuid = m_name2uuid[newName];
    auto& item = m_nodes[uuid];
    item->name = newName;   //update cache.

    if (m_subgNodes.find(oldName) != m_subgNodes.end()) {
        m_subgNodes.remove(oldName);
        m_subgNodes.insert(newName);
    }

    int row = m_uuid2Row[uuid];
    QModelIndex idx = createIndex(row, 0);
    emit dataChanged(idx, idx, QVector<int>{ QtRole::ROLE_NODE_NAME });
    emit nameUpdated(idx, oldName);
    zenoApp->graphsManager()->currentModel()->markDirty(true);
}

zeno::NodeData GraphModel::createNode(const QString& nodeCls, const QString& cate, const QPointF& pos)
{
    zeno::NodeData nodedata;
    nodedata.cls = nodeCls.toStdString();
    nodedata.uipos = { pos.x(), pos.y() };
    return _createNodeImpl(cate, nodedata, true);
}

void GraphModel::_appendNode(void* _spNode)
{
    zeno::NodeImpl* pNodeImpl = static_cast<zeno::NodeImpl*>(_spNode);
    int nRows = m_nodes.size();

    beginInsertRows(QModelIndex(), nRows, nRows);

    NodeItem* pItem = new NodeItem(this);
    pItem->init(this, pNodeImpl);
    pItem->params->setNodeIdx(createIndex(nRows, 0));

    const QString& name = QString::fromStdString(pNodeImpl->get_name());
    const QString& uuid = QString::fromStdString(pNodeImpl->get_uuid());

    if (pItem->optSubgraph.has_value())
        m_subgNodes.insert(name);

    m_row2uuid[nRows] = uuid;
    m_uuid2Row[uuid] = nRows;
    m_nodes.insert(uuid, pItem);
    m_name2uuid.insert(name, uuid);

    endInsertRows();

    zenoApp->graphsManager()->currentModel()->markDirty(true);
}

zeno::NodeData GraphModel::_createNodeImpl(const QString& cate, zeno::NodeData& nodedata, bool endTransaction)
{
    bool bEnableIoProc = zenoApp->graphsManager()->isInitializing() || zenoApp->graphsManager()->isImporting();
    if (bEnableIoProc)
        endTransaction = false;

    if (endTransaction)
    {
        auto currtPath = currentPath();
        AddNodeCommand* pCmd = new AddNodeCommand(cate, nodedata, currtPath);
        if (auto topLevelGraph = getTopLevelGraph(currtPath))
        {
            topLevelGraph->pushToplevelStack(pCmd);
            return pCmd->getNodeData();
        }
        return zeno::NodeData();
    }
    else {
        auto updateInputs = [](zeno::NodeData& nodedata, zeno::NodeImpl* spNode) {
            for (auto& tab : nodedata.customUi.inputPrims)
            {
                for (auto& group : tab.groups)
                {
                    for (auto& param : group.params)
                    {
                        spNode->update_param(param.name, param.defl);
                    }
                }
            }
        };

        auto spGraph = m_impl->m_wpCoreGraph;
        if (!spGraph)
            return zeno::NodeData();

        auto spNode = spGraph->createNode(nodedata.cls, nodedata.name, cate == "assets", nodedata.uipos);
        if (!spNode)
            return zeno::NodeData();

        zeno::NodeData node;

        if (zeno::isDerivedFromSubnetNodeName(nodedata.cls) || cate == "assets") {
            QString nodeName = QString::fromStdString(spNode->get_name());
            QString uuid = m_name2uuid[nodeName];
            ZASSERT_EXIT(m_nodes.find(uuid) != m_nodes.end(), zeno::NodeData());
            auto paramsM = m_nodes[uuid]->params;

            if (auto subnetNode = dynamic_cast<zeno::SubnetNode*>(spNode)) {
                //create input/output in subnet
                if (cate == "assets")
                {
                    const auto asset = zeno::getSession().assets->getAsset(nodedata.cls);
                    nodedata.customUi = asset.m_customui;
                }
                //zeno::ParamsUpdateInfo updateInfo;
                //zeno::parseUpdateInfo(nodedata.customUi, updateInfo);
                //paramsM->resetCustomUi(nodedata.customUi);
                //paramsM->batchModifyParams(updateInfo, true);

                if (nodedata.subgraph.has_value())
                {
                    for (auto& [name, nodedata] : nodedata.subgraph.value().nodes)
                    {
                        if (zeno::isDerivedFromSubnetNodeName(nodedata.cls)) {   //if is subnet, create recursively
                            QStringList cur = currentPath();
                            cur.append(QString::fromStdString(spNode->get_name()));
                            GraphModel* model = zenoApp->graphsManager()->getGraph(cur);
                            if (model)
                                model->_createNodeImpl(cate, nodedata, false);
                        }
                        else if (nodedata.cls == "SubInput" || nodedata.cls == "SubOutput") {   //dont create, just update subinput/output pos
                            auto ioNode = subnetNode->get_subgraph()->getNode(name);
                            if (ioNode)
                                ioNode->set_pos(nodedata.uipos);
                        }
                        else if (nodedata.asset.has_value()) {  //if is asset
                            spNode = subnetNode->get_subgraph()->createNode(nodedata.cls, name, true, {nodedata.uipos.first, nodedata.uipos.second});
                            if (spNode)
                                updateInputs(nodedata, spNode);
                        }
                        else {
                            spNode = subnetNode->get_subgraph()->createNode(nodedata.cls, name, false, {nodedata.uipos.first, nodedata.uipos.second});
                            if (spNode)
                                updateInputs(nodedata, spNode);
                        }
                    }
                    for (zeno::EdgeInfo oldLink : nodedata.subgraph.value().links) {
                        subnetNode->get_subgraph()->addLink(oldLink);
                    }
                }
                updateInputs(nodedata, spNode);
                node = spNode->exportInfo();
            }
        }
        else {
            updateInputs(nodedata, spNode);
            node = spNode->exportInfo();
        }

        return node;
    }
}

bool GraphModel::_removeNodeImpl(const QString& name, bool endTransaction)
{
    bool bEnableIoProc = zenoApp->graphsManager()->isInitializing() || zenoApp->graphsManager()->isImporting();
    if (bEnableIoProc)
        endTransaction = false;

    if (endTransaction)
    {
        if (m_name2uuid.find(name) != m_name2uuid.end() && m_nodes.find(m_name2uuid[name]) != m_nodes.end())
        {
            auto spNode = m_nodes[m_name2uuid[name]]->m_wpNode;
            if (spNode)
            {
                auto nodedata = spNode->exportInfo();
                auto currtPath = currentPath();
                RemoveNodeCommand* pCmd = new RemoveNodeCommand(nodedata, currtPath);
                if (auto topLevelGraph = getTopLevelGraph(currtPath))
                {
                    topLevelGraph->pushToplevelStack(pCmd);
                    return true;
                }
                //m_undoRedoStack->push(pCmd);
            }
        }
        return false;
    }
    else {
        //remove all related links
        NodeItem* item = m_nodes[m_name2uuid[name]];
        if (item)
        {
            PARAMS_INFO ioParams = item->params->getInputs();
            ioParams.insert(item->params->getOutputs());
            for (zeno::ParamPrimitive& paramInfo : ioParams)
            {
                for (zeno::EdgeInfo& edge: paramInfo.links)
                {
                    auto currtPath = currentPath();
                    LinkCommand* pCmd = new LinkCommand(false, edge, currentPath());
                    if (auto topLevelGraph = getTopLevelGraph(currtPath))
                        topLevelGraph->pushToplevelStack(pCmd);
                }
            }
        }

        auto spCoreGraph = m_impl->m_wpCoreGraph;
        ZASSERT_EXIT(spCoreGraph, false);
        if (spCoreGraph)
            return spCoreGraph->removeNode(name.toStdString());
        return false;
    }
}

void GraphModel::_addLink_apicall(const zeno::EdgeInfo& link, bool endTransaction)
{
    bool bEnableIoProc = zenoApp->graphsManager()->isInitializing() || zenoApp->graphsManager()->isImporting();
    if (bEnableIoProc)
        endTransaction = false;

    if (endTransaction)
    {
        LinkCommand* pCmd = new LinkCommand(true, link, currentPath());
        auto currtPath = currentPath();
        if (auto topLevelGraph = getTopLevelGraph(currtPath))
        {
            topLevelGraph->pushToplevelStack(pCmd);
        }
    }
    else {
        m_impl->m_wpCoreGraph->addLink(link);
    }
}

void GraphModel::_removeLinkImpl(const zeno::EdgeInfo& link, bool endTransaction)
{
    bool bEnableIoProc = zenoApp->graphsManager()->isInitializing() || zenoApp->graphsManager()->isImporting();
    if (bEnableIoProc)
        endTransaction = false;

    if (endTransaction)
    {
        LinkCommand* pCmd = new LinkCommand(false, link, currentPath());
        auto currtPath = currentPath();
        if (auto topLevelGraph = getTopLevelGraph(currtPath))
        {
            topLevelGraph->pushToplevelStack(pCmd);
        }
    }
    else {
        //emit to core data.
        m_impl->m_wpCoreGraph->removeLink(link);
    }
}

bool GraphModel::setModelData(const QModelIndex& index, const QVariant& newValue, int role)
{
    zeno::getSession().beginApiCall();
    zeno::scope_exit scope([=]() { zeno::getSession().endApiCall(); });

    const auto& oldVal = index.data(role);
    ModelDataCommand* pcmd = new ModelDataCommand(index, oldVal, newValue, role, currentPath());
    if (auto topLevelGraph = getTopLevelGraph(currentPath()))
    {
        topLevelGraph->pushToplevelStack(pcmd);
    }
    return true;
}

void GraphModel::_setViewImpl(const QModelIndex& idx, bool bOn, bool endTransaction)
{
    bool bEnableIoProc = zenoApp->graphsManager()->isInitializing() || zenoApp->graphsManager()->isImporting();
    if (bEnableIoProc)
        endTransaction = false;

    if (endTransaction)
    {
        auto currtPath = currentPath();
        NodeStatusCommand* pCmd = new NodeStatusCommand(zeno::View, idx.data(QtRole::ROLE_NODE_NAME).toString(), bOn, currtPath);
        if (auto topLevelGraph = getTopLevelGraph(currtPath))
            topLevelGraph->pushToplevelStack(pCmd);
    }
    else {
        auto spCoreGraph = m_impl->m_wpCoreGraph;
        ZASSERT_EXIT(spCoreGraph);
        NodeItem* item = m_nodes[m_row2uuid[idx.row()]];
        auto spCoreNode = item->m_wpNode;
        ZASSERT_EXIT(spCoreNode);
        spCoreNode->set_view(bOn);
    }
}

void GraphModel::_setNoCacheImpl(const QModelIndex& idx, bool bOn, bool endTransaction) {
    bool bEnableIoProc = zenoApp->graphsManager()->isInitializing() || zenoApp->graphsManager()->isImporting();
    if (bEnableIoProc)
        endTransaction = false;

    if (endTransaction) {
        auto currtPath = currentPath();
        NodeStatusCommand* pCmd = new NodeStatusCommand(zeno::Nocache, idx.data(QtRole::ROLE_NODE_NAME).toString(), bOn, currtPath);
        if (auto topLevelGraph = getTopLevelGraph(currtPath))
            topLevelGraph->pushToplevelStack(pCmd);
    }
    else {
        auto spCoreGraph = m_impl->m_wpCoreGraph;
        ZASSERT_EXIT(spCoreGraph);
        NodeItem* item = m_nodes[m_row2uuid[idx.row()]];
        auto spCoreNode = item->m_wpNode;
        ZASSERT_EXIT(spCoreNode);
        spCoreNode->set_nocache(bOn);
    }
}

void GraphModel::clearNodeObjs(const QModelIndex& nodeIdx) {
    auto spCoreGraph = m_impl->m_wpCoreGraph;
    ZASSERT_EXIT(spCoreGraph);
    auto iter = m_row2uuid.find(nodeIdx.row());
    ZASSERT_EXIT(iter != m_row2uuid.end());
    NodeItem* item = m_nodes[m_row2uuid[nodeIdx.row()]];
    zeno::NodeImpl* spNode = item->m_wpNode;
    spNode->clearCalcResults();
}

void GraphModel::clearSubnetObjs(const QModelIndex& nodeIdx) {
    auto spCoreGraph = m_impl->m_wpCoreGraph;
    ZASSERT_EXIT(spCoreGraph);
    auto iter = m_row2uuid.find(nodeIdx.row());
    ZASSERT_EXIT(iter != m_row2uuid.end());
    NodeItem* item = m_nodes[m_row2uuid[nodeIdx.row()]];
    zeno::NodeImpl* spNode = item->m_wpNode;
    ZASSERT_EXIT(spNode->nodeType() == zeno::Node_SubgraphNode ||
        spNode->nodeType() == zeno::Node_AssetInstance);
    zeno::SubnetNode* subnetnode = static_cast<zeno::SubnetNode*>(spNode);
    subnetnode->cleanInternalCaches();
}

void GraphModel::_setClearSubnetImpl(const QModelIndex& idx, bool bOn, bool endTransaction) {
    bool bEnableIoProc = zenoApp->graphsManager()->isInitializing() || zenoApp->graphsManager()->isImporting();
    if (bEnableIoProc)
        endTransaction = false;

    if (endTransaction) {
        auto currtPath = currentPath();
        NodeStatusCommand* pCmd = new NodeStatusCommand(zeno::ClearSbn, idx.data(QtRole::ROLE_NODE_NAME).toString(), bOn, currtPath);
        if (auto topLevelGraph = getTopLevelGraph(currtPath))
            topLevelGraph->pushToplevelStack(pCmd);
    }
    else {
        auto spCoreGraph = m_impl->m_wpCoreGraph;
        ZASSERT_EXIT(spCoreGraph);
        NodeItem* item = m_nodes[m_row2uuid[idx.row()]];
        auto spCoreNode = item->m_wpNode;
        ZASSERT_EXIT(spCoreNode);
        zeno::SubnetNode* subnetnode = static_cast<zeno::SubnetNode*>(spCoreNode);
        subnetnode->set_clearsubnet(bOn);
    }
}

void GraphModel::_setByPassImpl(const QModelIndex& idx, bool bOn, bool endTransaction)
{
    bool bEnableIoProc = zenoApp->graphsManager()->isInitializing() || zenoApp->graphsManager()->isImporting();
    if (bEnableIoProc)
        endTransaction = false;

    if (endTransaction)
    {
        auto currtPath = currentPath();
        NodeStatusCommand* pCmd = new NodeStatusCommand(zeno::ByPass, idx.data(QtRole::ROLE_NODE_NAME).toString(), bOn, currtPath);
        if (auto topLevelGraph = getTopLevelGraph(currtPath))
            topLevelGraph->pushToplevelStack(pCmd);
    }
    else {
        auto spCoreGraph = m_impl->m_wpCoreGraph;
        ZASSERT_EXIT(spCoreGraph);
        NodeItem* item = m_nodes[m_row2uuid[idx.row()]];
        auto spCoreNode = item->m_wpNode;
        ZASSERT_EXIT(spCoreNode);
        spCoreNode->set_bypass(bOn);
    }
}

void GraphModel::appendSubgraphNode(QString name, QString cls, NODE_DESCRIPTOR desc, GraphModel* subgraph, const QPointF& pos)
{
    //TODO:
#if 0
    int nRows = m_nodes.size();
    beginInsertRows(QModelIndex(), nRows, nRows);

    NodeItem* pItem = new NodeItem(this);
    pItem->setParent(this);
    pItem->name = name;
    pItem->name = cls;
    pItem->pos = pos;
    pItem->params = new ParamsModel(desc, pItem);
    pItem->pSubgraph = subgraph;
    subgraph->setParent(pItem);

    m_row2name[nRows] = name;
    m_name2Row[name] = nRows;
    m_nodes.insert(name, pItem);

    endInsertRows();
    pItem->params->setNodeIdx(createIndex(nRows, 0));
#endif
}

bool GraphModel::removeNode(const QString& name)
{
    if (m_name2uuid.find(name) != m_name2uuid.end() && m_nodes.find(m_name2uuid[name]) != m_nodes.end()) {
        return _removeNodeImpl(name, true);
    }
    return false;
}

void GraphModel::setView(const QModelIndex& idx, bool bOn)
{
    _setViewImpl(idx, bOn, true);
}

void GraphModel::setBypass(const QModelIndex& idx, bool bOn)
{
    _setByPassImpl(idx, bOn, true);
}

void GraphModel::setNocache(const QModelIndex& idx, bool bOn) {
    _setNoCacheImpl(idx, bOn, true);
}

void GraphModel::setClearSubnet(const QModelIndex& idx, bool bOn) {
    _setClearSubnetImpl(idx, bOn, true);
}

QString GraphModel::updateNodeName(const QModelIndex& idx, QString newName)
{
    auto spCoreGraph = m_impl->m_wpCoreGraph;
    ZASSERT_EXIT(spCoreGraph, "");

    std::string oldName = idx.data(QtRole::ROLE_NODE_NAME).toString().toStdString();
    newName = QString::fromStdString(spCoreGraph->updateNodeName(oldName, newName.toStdString()));
    return newName;
}

void GraphModel::updateSocketValue(const QModelIndex& nodeidx, const QString socketName, const QVariant newValue)
{
    m_impl->m_wpCoreGraph->hasNode("");
    if (ParamsModel* paramModel = params(nodeidx))
    {
        const QModelIndex& socketIdx = paramModel->paramIdx(socketName, true);
        paramModel->setData(socketIdx, newValue, QtRole::ROLE_PARAM_VALUE);
    }
}

QHash<int, QByteArray> GraphModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[QtRole::ROLE_CLASS_NAME] = "classname";
    roles[QtRole::ROLE_NODE_NAME] = "name";
    //roles[QtRole::ROLE_NODE_UISTYLE] = "uistyle";
    roles[QtRole::ROLE_PARAMS] = "params";
    roles[QtRole::ROLE_LINKS] = "linkModel";
    roles[QtRole::ROLE_OBJPOS] = "pos";
    roles[QtRole::ROLE_SUBGRAPH] = "subgraph";
    roles[QtRole::ROLE_NODE_LOCKED] = "locked";
    return roles;
}

void GraphModel::_clear()
{
    while (rowCount() > 0)
    {
        const QString& nodeName = index(0, 0).data(QtRole::ROLE_NODE_NAME).toString();
        removeNode(nodeName);
    }
}

bool GraphModel::removeRows(int row, int count, const QModelIndex& parent)
{
    //this is a private impl method, called by callback function.
    beginRemoveRows(parent, row, row);

    QString id = m_row2uuid[row];
    NodeItem* pItem = m_nodes[id];
    const QString& name = pItem->getName();

    for (int r = row + 1; r < rowCount(); r++)
    {
        const QString& id_ = m_row2uuid[r];
        m_row2uuid[r - 1] = id_;
        m_uuid2Row[id_] = r - 1;
    }

    m_row2uuid.remove(rowCount() - 1);
    m_uuid2Row.remove(id);
    m_nodes.remove(id);
    m_name2uuid.remove(name);

    if (m_subgNodes.find(name) != m_subgNodes.end())
        m_subgNodes.remove(name);

    delete pItem;

    endRemoveRows();

    emit nodeRemoved(name);
    return true;
}

void GraphModel::syncToAssetsInstance_customui(const QString& assetsName, zeno::ParamsUpdateInfo info, const zeno::CustomUI& customui)
{
    QModelIndexList results = match(QModelIndex(), QtRole::ROLE_CLASS_NAME, assetsName);
    for (QModelIndex res : results) {
        zeno::NodeType type = (zeno::NodeType)res.data(QtRole::ROLE_NODETYPE).toInt();
        if (!res.data(QtRole::ROLE_NODE_LOCKED).toBool())
            continue;
        if (type == zeno::Node_AssetInstance || type == zeno::Node_AssetReference) {
            ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(res.data(QtRole::ROLE_PARAMS));
            ZASSERT_EXIT(paramsM);
            paramsM->resetCustomUi(customui);
            paramsM->batchModifyParams(info);
        }
    }

    const QStringList& path = currentPath();
    if (!path.isEmpty() && path[0] != "main") {
        return;
    }

    syncToAssetsInstance(assetsName);

    for (QString subgnode : m_subgNodes) {
        ZASSERT_EXIT(m_name2uuid.find(subgnode) != m_name2uuid.end());
        QString uuid = m_name2uuid[subgnode];
        ZASSERT_EXIT(m_nodes.find(uuid) != m_nodes.end());
        GraphModel* pSubgM = m_nodes[uuid]->optSubgraph.value();
        ZASSERT_EXIT(pSubgM);
        pSubgM->syncToAssetsInstance_customui(assetsName, info, customui);
    }
}

void GraphModel::resetAssetAndLock(const QModelIndex& assetNode)
{
    auto spCoreGraph = m_impl->m_wpCoreGraph;
    ZASSERT_EXIT(spCoreGraph);
    auto iter = m_row2uuid.find(assetNode.row());
    ZASSERT_EXIT(iter != m_row2uuid.end());
    NodeItem* item = m_nodes[m_row2uuid[assetNode.row()]];
    zeno::NodeImpl* spNode = item->m_wpNode;
    zeno::SubnetNode* subnetnode = static_cast<zeno::SubnetNode*>(spNode);
    std::string assetname = item->cls.toStdString();

    //只需要把最终要确定的参数加到updateInfo即可
    auto& assets = zeno::getSession().assets;
    const zeno::Asset& asset = assets->getAsset(assetname);
    ParamsModel* paramsM = item->params;
    PARAMS_INFO oldinputs = paramsM->getInputs();
    PARAMS_INFO oldoutputs = paramsM->getOutputs();

    zeno::ParamsUpdateInfo updateInfo;
    for (const zeno::ParamObject& param : asset.object_inputs) {
        QString qsName = QString::fromStdString(param.name);
        zeno::ParamUpdateInfo info;
        info.param = param;
        //看param在oldinputs是否存在
        if (oldinputs.find(qsName) != oldinputs.end()) {
            info.oldName = param.name;  //更名的情况估计会有问题，但考虑到可能用的不多，就这么处理先
        }
        updateInfo.emplace_back(std::move(info));
    }
    for (const zeno::ParamPrimitive& param : asset.primitive_inputs) {
        QString qsName = QString::fromStdString(param.name);
        zeno::ParamUpdateInfo info;
        info.param = param;
        //看param在oldinputs是否存在
        if (oldinputs.find(qsName) != oldinputs.end()) {
            info.oldName = param.name;
        }
        updateInfo.emplace_back(std::move(info));
    }
    for (const zeno::ParamPrimitive& param : asset.primitive_outputs) {
        QString qsName = QString::fromStdString(param.name);
        zeno::ParamUpdateInfo info;
        info.param = param;
        //看param在oldinputs是否存在
        if (oldoutputs.find(qsName) != oldoutputs.end()) {
            info.oldName = param.name;
        }
        updateInfo.emplace_back(std::move(info));
    }
    for (const zeno::ParamObject& param : asset.object_outputs) {
        QString qsName = QString::fromStdString(param.name);
        zeno::ParamUpdateInfo info;
        info.param = param;
        //看param在oldinputs是否存在
        if (oldoutputs.find(qsName) != oldoutputs.end()) {
            info.oldName = param.name;
        }
        updateInfo.emplace_back(std::move(info));
    }

    paramsM->resetCustomUi(asset.m_customui);
    paramsM->batchModifyParams(updateInfo);

    GraphModel* oldSubgraphM = item->optSubgraph.value();
    //要清掉旧的图（但不包括容器本身，只是删节点和边）
    oldSubgraphM->clear();

    std::shared_ptr<zeno::Graph> newSubgraph = assets->forkAssetGraph(asset.sharedGraph, spNode);
    subnetnode->init_graph(newSubgraph);
    oldSubgraphM->m_impl->m_wpCoreGraph = newSubgraph.get();
    oldSubgraphM->registerCoreNotify();

    //重新把点和边加回去
    std::map<std::string, zeno::NodeImpl*> nodes;
    nodes = newSubgraph->getNodes();
    for (auto& [name, node] : nodes) {
        oldSubgraphM->_appendNode(node);
    }
    oldSubgraphM->_initLink();

    setData(assetNode, true, QtRole::ROLE_NODE_LOCKED);

    //得把事务清掉，因为这里没法支持undo/redo
    GraphModel* mainG = zenoApp->graphsManager()->getGraph({ "main" });
    mainG->m_undoRedoStack.value()->clear();
}

void GraphModel::syncAssetInst(const QModelIndex& assetNode) {
    
    if (!assetNode.isValid()) return;

    bool isLocked = assetNode.data(QtRole::ROLE_NODE_LOCKED).toBool();
    ZASSERT_EXIT(!isLocked);

    QString assetName = assetNode.data(QtRole::ROLE_CLASS_NAME).toString();
    NodeItem* item = m_nodes[m_row2uuid[assetNode.row()]];
    zeno::NodeImpl* spNode = item->m_wpNode;
    zeno::SubnetNode* subnetnode = static_cast<zeno::SubnetNode*>(spNode);

    auto& assets = zeno::getSession().assets;
    AssetsModel* assetsM = zenoApp->graphsManager()->assetsModel();
    GraphModel* pAssetGraph = assetsM->getAssetGraph(assetName);
    GraphModel* mainG = zenoApp->graphsManager()->getGraph({ "main" });

    //先移除assetModel所有的节点
    pAssetGraph->clear();

    //fork一份assetNode的graph，设置到asset上
    std::shared_ptr<zeno::Graph> newSharedAsset = assets->syncInstToAssets(spNode);
    zeno::CustomUI newUi = subnetnode->export_customui();
    assets->updateAssetInfo(assetName.toStdString(), newSharedAsset, newUi);
    //uimodel上增加节点和边
    std::map<std::string, zeno::NodeImpl*> nodes;
    nodes = newSharedAsset->getNodes();
    for (auto& [name, node] : nodes) {
        pAssetGraph->_appendNode(node);
    }
    pAssetGraph->_initLink();

    //同步到其他的instance
    mainG->syncToAssetsInstance(assetName);

    //有可能改了asset的customui，而别的asset引用了这个修改后的asset，而且有节点参数ui，就意味着
    //所有asset也得同步一下这个


    //最后给当前assetNode上锁
    setData(assetNode, true, QtRole::ROLE_NODE_LOCKED);

    //得把事务清掉，因为这里没法支持undo/redo
    mainG->m_undoRedoStack.value()->clear();
}

void GraphModel::syncToAssetsInstance(const QString& assetsName)
{
    for (const QString & name : m_subgNodes)
    {
        ZASSERT_EXIT(m_name2uuid.find(name) != m_name2uuid.end());
        QString uuid = m_name2uuid[name];
        ZASSERT_EXIT(m_nodes.find(uuid) != m_nodes.end());
        GraphModel* pSubgM = m_nodes[uuid]->optSubgraph.value();
        ZASSERT_EXIT(pSubgM);
        if (assetsName == m_nodes[uuid]->cls)
        {
            //TO DO: compare diff
            if (!pSubgM->isLocked())
                continue;
            while (pSubgM->rowCount() > 0)
            {
                const QString& nodeName = pSubgM->index(0, 0).data(QtRole::ROLE_NODE_NAME).toString();
                pSubgM->removeNode(nodeName);
            }
            auto spNode = m_nodes[uuid]->m_wpNode;
            auto spSubnetNode = dynamic_cast<zeno::SubnetNode*>(spNode);
            if (spSubnetNode) {
                auto& assetsMgr = zeno::getSession().assets;
                assetsMgr->updateAssetInstance(assetsName.toStdString(), spSubnetNode);
                //pSubgM->updateAssetInstance(spSubnetNode->subgraph.get());
                {
                    auto spGraph = spSubnetNode->get_subgraph();
                    pSubgM->m_impl->m_wpCoreGraph = spGraph;
                    pSubgM->registerCoreNotify();
                    for (auto& [name, node] : spGraph->getNodes()) {
                        pSubgM->_appendNode(node);
                    }
                    pSubgM->_initLink();
                }
                spNode->mark_dirty(true);
            }
        }
        else
        {
            pSubgM->syncToAssetsInstance(assetsName);
        }
    }
}

void GraphModel::updateParamName(QModelIndex nodeIdx, int row, QString newName)
{
    NodeItem* item = m_nodes[m_row2uuid[nodeIdx.row()]];
    QModelIndex paramIdx = item->params->index(row, 0);
    item->params->setData(paramIdx, newName, QtRole::ROLE_PARAM_NAME);
}

void GraphModel::removeParam(QModelIndex nodeIdx, int row)
{
    NodeItem* item = m_nodes[m_row2uuid[nodeIdx.row()]];
    item->params->removeRow(row);
}

ParamsModel* GraphModel::params(QModelIndex nodeIdx)
{
    NodeItem* item = m_nodes[m_row2uuid[nodeIdx.row()]];
    return item->params;
}

GraphModel* GraphModel::subgraph(QModelIndex nodeIdx) {
    NodeItem* item = m_nodes[m_row2uuid[nodeIdx.row()]];
    if (item->optSubgraph.has_value()) {
        return item->optSubgraph.value();
    }
    return nullptr;
}

GraphsTreeModel* GraphModel::treeModel() const {
    return m_pTree;
}

bool GraphModel::isLocked() const
{
    if (m_graphName == "main" || !m_impl->m_parentM) { return false; }

    if (m_impl->m_parentM->isLocked()) { return true; }

    //观察当前图是不是资产图，以及找到上层的资产节点，看是不是上锁了
    zeno::NodeImpl* parSubnetNode = m_impl->m_wpCoreGraph->getParentSubnetNode();
    if (!parSubnetNode) return false;
    ZASSERT_EXIT(parSubnetNode, false);
    if (zeno::Node_AssetInstance == parSubnetNode->nodeType()) {
        return parSubnetNode->is_locked();
    }
    return false;
}

QString GraphModel::name2uuid(const QString& name) {
    ZASSERT_EXIT(m_name2uuid.find(name) != m_name2uuid.end(), "");
    return m_name2uuid[name];
}

QStringList GraphModel::pasteNodes(const zeno::NodesData& nodes, const zeno::LinksData& links, const QPointF& pos)
{
    QStringList newnode_names;
    if (nodes.empty())
        return newnode_names;

    std::map<std::string, std::string> old2new;
    QPointF offset = pos - QPointF(nodes.begin()->second.uipos.first, nodes.begin()->second.uipos.second);
    unRegisterCoreNotify();
    for (auto [src_name, node] : nodes) {
        bool bAsset = node.asset.has_value();
        auto spNode = m_impl->m_wpCoreGraph->createNode(node.cls, src_name, bAsset, { offset.x(), offset.y() }, true);
        node.name = spNode->get_name();
        spNode->init(node);
        auto pos = spNode->get_pos();
        spNode->set_pos({ pos.first + offset.x(), pos.second + offset.y()});
        old2new[src_name] = node.name;
        _appendNode(spNode);
        newnode_names.append(QString::fromStdString(node.name));
    }
    registerCoreNotify();
    //import edges
    for (auto link : links) {
        if (old2new.find(link.outNode) == old2new.end() || old2new.find(link.inNode) == old2new.end())
            continue;
        link.inNode = old2new[link.inNode];
        link.outNode = old2new[link.outNode];
        addLink(link);
    }
    return newnode_names;
}

GraphModel* GraphModel::getTopLevelGraph(const QStringList& currentPath)
{
    return zenoApp->graphsManager()->getGraph({ currentPath[0] });
}
