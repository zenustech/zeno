#include "graphstreemodel.h"
#include "uicommon.h"
#include "variantptr.h"
#include "descriptors.h"
#include <zeno/core/common.h>


GraphsTreeModel::GraphsTreeModel(QObject* parent)
    : QAbstractItemModel(parent)
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
}

QModelIndex GraphsTreeModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= rowCount(parent))
        return QModelIndex();

    if (parent.isValid()) {
        GraphModel* pSubgraph = parent.data(ROLE_SUBGRAPH).value<GraphModel*>();
        Q_ASSERT(pSubgraph);
        return createIndex(row, column, pSubgraph);
    }
    else {
        return createIndex(row, column, nullptr);
    }
}

QModelIndex GraphsTreeModel::parent(const QModelIndex& child) const
{
    if (!child.isValid())
        return QModelIndex();

    if (child.internalId() == 0) {  //main item on root.
        return QModelIndex();
    }

    QModelIndex innerChild = innerIndex(child);
    auto pModel = innerChild.model();
    if (!pModel) {
        return QModelIndex();
    }
    if (auto pItem = qobject_cast<NodeItem*>(pModel->parent()))
    {
        if (auto parentModel = qobject_cast<GraphModel*>(pItem->parent()))
        {
            int row = parentModel->indexFromId(pItem->getName());
            return createIndex(row, 0, parentModel);
        }
    }
    return createIndex(0, 0);   //main item
}

int GraphsTreeModel::rowCount(const QModelIndex& parent) const
{
    if (!parent.isValid()) {
        //main item
        return 1;
    }
    else {
        GraphModel* pSubgraph = parent.data(ROLE_SUBGRAPH).value<GraphModel*>();
        return pSubgraph ? pSubgraph->rowCount() : 0;
    }
}

int GraphsTreeModel::columnCount(const QModelIndex& parent) const
{
    return 1;
}

bool GraphsTreeModel::hasChildren(const QModelIndex& parent) const
{
    if (!parent.isValid()) {
        return true;    //the only child is `main` item.
    }
    else {
        GraphModel* pSubgraph = parent.data(ROLE_SUBGRAPH).value<GraphModel*>();
        return pSubgraph ? (pSubgraph->rowCount() > 0) : false;
    }
    return false;
}

QModelIndex GraphsTreeModel::innerIndex(const QModelIndex& treeIdx) const
{
    if (!treeIdx.isValid())
        return QModelIndex();

    //tree项的行号和图模型的行号是一致的。
    GraphModel* ownerModel = static_cast<GraphModel*>(treeIdx.internalPointer());
    Q_ASSERT(ownerModel);
    return ownerModel->index(treeIdx.row(), 0);
}

QVariant GraphsTreeModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (index.internalId() == 0) {
        //main item
        if (Qt::DisplayRole == role || ROLE_NODE_NAME == role || ROLE_CLASS_NAME == role) {
            return "main";
        }
        else if (ROLE_SUBGRAPH == role) {   //相当于子图节点那样，main可以看作最根部的子图节点
            return QVariant::fromValue(m_main);
        }
        else if (ROLE_OBJPATH == role) {
            return "/main";
        }
        return QVariant();
    }

    QModelIndex innerIdx = innerIndex(index);
    return innerIdx.data(role);
}

bool GraphsTreeModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    QModelIndex innerIdx = innerIndex(index);
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(innerIdx.model());
    return pModel ? pModel->setData(innerIdx, value, role) : false;
}

QModelIndexList GraphsTreeModel::match(const QModelIndex& start, int role,
    const QVariant& value, int hits,
    Qt::MatchFlags flags) const
{
    return QModelIndexList();
}

QHash<int, QByteArray> GraphsTreeModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[ROLE_CLASS_NAME] = "class";
    roles[ROLE_NODE_NAME] = "name";
    roles[ROLE_PARAMS] = "params";
    roles[ROLE_LINKS] = "linkModel";
    roles[ROLE_OBJPOS] = "pos";
    roles[ROLE_SUBGRAPH] = "subgraph";
    return roles;
}

void GraphsTreeModel::onGraphRowsInserted(const QModelIndex& parent, int first, int last)
{
    GraphModel* pGraphM = qobject_cast<GraphModel*>(sender());
    if (pGraphM)
    {
        QStringList graphPath = pGraphM->currentPath();
        QModelIndex treeParentItem = getIndexByPath(graphPath);
        emit layoutChanged({ treeParentItem });
    }
}

void GraphsTreeModel::onNameUpdated(const QModelIndex& nodeIdx, const QString& oldName)
{
    GraphModel* pGraphM = qobject_cast<GraphModel*>(sender());
    if (pGraphM)
    {
        QStringList nodePath = nodeIdx.data(ROLE_OBJPATH).toStringList();
        QModelIndex nodeIdx = getIndexByPath(nodePath);
        emit dataChanged(nodeIdx, nodeIdx, { Qt::DisplayRole, ROLE_NODE_NAME });
    }
}

void GraphsTreeModel::onGraphRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    //GraphModel* pGraphM = qobject_cast<GraphModel*>(sender());
    //QString graphPath = pGraphM->currentPath();
    //QModelIndex treeParentItem = getIndexByPath(graphPath);
    //emit layoutAboutToBeChanged({ treeParentItem });
}

void GraphsTreeModel::onGraphRowsRemoved(const QModelIndex& parent, int first, int last)
{
    GraphModel* pGraphM = qobject_cast<GraphModel*>(sender());
    if (pGraphM) {
        QStringList graphPath = pGraphM->currentPath();
        QModelIndex treeParentItem = getIndexByPath(graphPath);
        emit layoutChanged({ treeParentItem });
    }
}

int GraphsTreeModel::depth(const QModelIndex& index) const
{
    int count = 0;
    auto anchestor = index;
    if (!index.isValid()) {
        return 0;
    }
    while (anchestor.parent().isValid()) {
        anchestor = anchestor.parent();
        ++count;
    }
    return count;
}

GraphModel* GraphsTreeModel::graph(const QModelIndex& index) const
{
    if (index.isValid() && index.internalId() == 0)
        return nullptr;

    GraphModel* ownerModel = static_cast<GraphModel*>(index.internalPointer());
    Q_ASSERT(ownerModel);
    return ownerModel;
}

QString GraphsTreeModel::name(const QModelIndex& index) const
{
    return index.data(ROLE_NODE_NAME).toString();
}

//! Clear the model.
void GraphsTreeModel::clear()
{
    emit layoutAboutToBeChanged();
    beginResetModel();
    //delete m_main;
    //m_main = new GraphModel("main", this);
    endResetModel();
    emit layoutChanged();
    emit modelClear();
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
    }
    return nullptr;
}

QModelIndex GraphsTreeModel::getIndexByUuidPath(const zeno::ObjPath& objPath)
{
    if (!m_main)
        return QModelIndex();
    return m_main->indexFromUuidPath(objPath);
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
                pGraphM = innerIdx.data(ROLE_SUBGRAPH).value<GraphModel*>();
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
