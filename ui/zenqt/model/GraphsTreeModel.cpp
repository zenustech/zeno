#include "GraphsTreeModel.h"
#include "uicommon.h"
#include "graphsmanager.h"
#include "zenoapplication.h"
#include "variantptr.h"
#include <zeno/core/common.h>
#include "assetsmodel.h"
#include "zassert.h"


static void initGraphItems(QStandardItem* root, GraphModel* model)
{
    for (int r = 0; r < model->rowCount(); r++)
    {
        QPersistentModelIndex idx = model->index(r, 0);
        const QString& name = idx.data(QtRole::ROLE_NODE_NAME).toString();
        QStandardItem* pItem = new QStandardItem(name);
        //QVariant::PersistentModelIndex
        pItem->setData(idx, Qt::UserRole + 1);
        QVariant val = idx.data(QtRole::ROLE_SUBGRAPH);
        if (!val.isNull()) {
            GraphModel* pSubgGraphM = idx.data(QtRole::ROLE_SUBGRAPH).value<GraphModel*>();
            initGraphItems(pItem, pSubgGraphM);
        }
        root->appendRow(pItem);
    }
}

static QStandardItem* findGraphItem(QStandardItem* parentItem, QStringList graph_path) {
    if (graph_path.isEmpty()) {
        return parentItem;
    }
    const QString& nodename = graph_path.front();
    graph_path.pop_front();
    for (int r = 0; r < parentItem->rowCount(); r++) {
        QStandardItem* item = parentItem->child(r);
        if (item->text() == nodename) {
            return findGraphItem(item, graph_path);
        }
    }
    return nullptr;
}


GraphsTreeModel::GraphsTreeModel(QObject* parent)
    : _base(parent)
    , m_main(nullptr)
    , m_dirty(false)
{
}

GraphsTreeModel::~GraphsTreeModel()
{
}

void GraphsTreeModel::init(GraphModel* mainModel)
{
    m_main = mainModel;
    QStandardItem* main_item = new QStandardItem("main");
    initGraphItems(main_item, mainModel);
    appendRow(main_item);
    emit layoutChanged({ main_item->index() });
}

QHash<int, QByteArray> GraphsTreeModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[QtRole::ROLE_CLASS_NAME] = "class";
    roles[QtRole::ROLE_NODE_NAME] = "name";
    roles[QtRole::ROLE_PARAMS] = "params";
    roles[QtRole::ROLE_LINKS] = "linkModel";
    roles[QtRole::ROLE_OBJPOS] = "pos";
    roles[QtRole::ROLE_SUBGRAPH] = "subgraph";
    return roles;
}

QStandardItem* GraphsTreeModel::initNodeItem(const QModelIndex& nodeidx) const
{
    QString nodename = nodeidx.data(QtRole::ROLE_NODE_NAME).toString();
    QStandardItem* pItem = new QStandardItem(nodename);
    pItem->setData(nodeidx, Qt::UserRole + 1);

    zeno::NodeType type = static_cast<zeno::NodeType>(nodeidx.data(QtRole::ROLE_NODETYPE).toInt());
    if (type == zeno::Node_AssetInstance ||
        type == zeno::Node_AssetReference ||
        type == zeno::Node_SubgraphNode)
    {
        GraphModel* pSubgraph = nodeidx.data(QtRole::ROLE_SUBGRAPH).value<GraphModel*>();
        for (int r = 0; r < pSubgraph->rowCount(); r++)
        {
            QStandardItem* pChildItem = initNodeItem(pSubgraph->index(r));
            pItem->appendRow(pChildItem);
        }
    }
    return pItem;
}

void GraphsTreeModel::onGraphRowsInserted(const QModelIndex& parent, int first, int last)
{
    GraphModel* pGraphM = qobject_cast<GraphModel*>(sender());
    if (pGraphM)
    {
        QStringList graphPath = pGraphM->currentPath();
        ZASSERT_EXIT(!graphPath.isEmpty());
        QStandardItem* mainItem = itemFromIndex(this->index(0, 0));
        graphPath.pop_front();  //"main"
        QStandardItem* graphItem = findGraphItem(mainItem, graphPath);
        ZASSERT_EXIT(graphItem);

        QPersistentModelIndex newNodeIdx = pGraphM->index(first);
        ZASSERT_EXIT(newNodeIdx.isValid());
        QStandardItem* pItem = initNodeItem(newNodeIdx);
        graphItem->appendRow(pItem);

        emit layoutChanged({ graphItem->index() });
    }
}

void GraphsTreeModel::onNameUpdated(const QModelIndex& nodeIdx, const QString& oldName)
{
    GraphModel* pGraphM = qobject_cast<GraphModel*>(sender());
    if (pGraphM)
    {
        QStringList graphPath = pGraphM->currentPath();
        ZASSERT_EXIT(!graphPath.isEmpty());
        QStandardItem* mainItem = itemFromIndex(this->index(0, 0));
        graphPath.pop_front();  //"main"
        QStandardItem* graphItem = findGraphItem(mainItem, graphPath);
        ZASSERT_EXIT(graphItem);

        const QString& newName = nodeIdx.data(QtRole::ROLE_NODE_NAME).toString();
        //目前不建立索引，直接顺序遍历，待遇到性能问题再行优化
        for (int i = 0; i < graphItem->rowCount(); i++) {
            QStandardItem* pItem = graphItem->child(i);
            if (pItem->text() == oldName) {
                pItem->setText(newName);
                break;
            }
        }
    }
}

void GraphsTreeModel::onGraphRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    //有可能tree已经提前被clear了，比如关闭工程的时候
    if (rowCount() == 0) {
        return;
    }

    GraphModel* pGraphM = qobject_cast<GraphModel*>(sender());
    if (pGraphM) {
        QStringList graphPath = pGraphM->currentPath();
        ZASSERT_EXIT(!graphPath.isEmpty());
        QStandardItem* mainItem = itemFromIndex(this->index(0, 0));
        graphPath.pop_front();  //"main"
        QStandardItem* graphItem = findGraphItem(mainItem, graphPath);
        ZASSERT_EXIT(graphItem);

        graphItem->removeRow(first);
        //emit layoutChanged({ graphItem->index() }); //树比较大的时候有性能损耗
    }
}

void GraphsTreeModel::onGraphRowsRemoved(const QModelIndex& parent, int first, int last)
{
    GraphModel* pGraphM = qobject_cast<GraphModel*>(sender());
    if (pGraphM) {
        QStringList graphPath = pGraphM->currentPath();
    }
}

int GraphsTreeModel::depth(const QModelIndex& index) const
{
    if (!index.isValid())
        return 0;

    QStandardItem* pItem = itemFromIndex(index);
    int count = 0;
    while (pItem)
    {
        pItem = pItem->parent();
        count++;
    }
    return count;
}

GraphModel* GraphsTreeModel::graph(const QModelIndex& index) const
{
    if (!index.isValid())
        return nullptr;

    QStandardItem* pItem = itemFromIndex(index);
    QStandardItem* parentItem = pItem->parent();
    if (parentItem == nullptr && pItem->text() == tr("main")) {
        return m_main;
    }

    QPersistentModelIndex idx = pItem->data(Qt::UserRole + 1).value<QPersistentModelIndex>();
    QAbstractItemModel* ownerModel = const_cast<QAbstractItemModel*>(idx.model());
    return qobject_cast<GraphModel*>(ownerModel);
}

QString GraphsTreeModel::name(const QModelIndex& index) const
{
    return index.data(QtRole::ROLE_NODE_NAME).toString();
}

//! Clear the model.
void GraphsTreeModel::clear()
{
    //remove main
    removeRow(0);
    QModelIndex rootIndex = invisibleRootItem()->index();
    m_main = nullptr;
    emit modelClear();
    m_dirty = false;
}

/*!
*  Return the root item to the QML Side.
*  This method is not meant to be used in client code.
*/
QModelIndex GraphsTreeModel::rootIndex()
{
    return {};
}

GraphModel* GraphsTreeModel::getGraphByPath(const QStringList& objPath)
{
    if (!m_main)
        return nullptr;

    QStringList items = objPath;
    if (items.empty()) {
        //TODO: ASSETS
        return nullptr;
    }
    else {
        if (items[0] == "main") {
            items.removeAt(0);
            return m_main->getGraphByPath(items);
        }
        else {
            //资产的情况
            AssetsModel* assets = zenoApp->graphsManager()->assetsModel();
            GraphModel* pAssetGraph = assets->getAssetGraph(items[0]);
            items.removeAt(0);
            return pAssetGraph->getGraphByPath(items);
        }
    }
    return nullptr;
}

QModelIndex GraphsTreeModel::getIndexByUuidPath(const zeno::ObjPath& objPath)
{
    return zenoApp->graphsManager()->getNodeIndexByUuidPath(QString::fromStdString(objPath));
}

QModelIndex GraphsTreeModel::getIndexByPath(const QStringList& objPath)
{
    QStringList items = objPath;
    if (items.empty()) {
        return QModelIndex();
    }
    else {
        if (items[0] == "main") {
            items.removeAt(0);
            GraphModel* pGraphM = m_main;
            QModelIndex curNode = createIndex(0, 0);
            //["main", "aaa", "bbb", "ccc", "createcube1"]
            while (!items.isEmpty()) {
                QString node = items[0];
                if (!pGraphM)
                    break;

                QModelIndex innerIdx = pGraphM->indexFromName(node);
                curNode = createIndex(innerIdx.row(), 0, pGraphM);
                items.removeAt(0);
                pGraphM = innerIdx.data(QtRole::ROLE_SUBGRAPH).value<GraphModel*>();
            }
            return curNode;
        }
    }
    return QModelIndex();
}

bool GraphsTreeModel::isDirty() const {
    return m_dirty;
}

void GraphsTreeModel::clearDirty() {
    if (m_dirty)
    {
        m_dirty = false;
        emit dirtyChanged();
    }
}

QList<SEARCH_RESULT> GraphsTreeModel::search(const QString& content, int searchType, int searchOpts) const
{
    return m_main->search(content, SearchType(searchType), SearchOpt(searchOpts));
}

QList<SEARCH_RESULT> GraphsTreeModel::searchByUuidPath(const zeno::ObjPath& uuidPath)
{
    return m_main->searchByUuidPath(uuidPath);
}
