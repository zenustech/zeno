#include "GraphModel.h"
#include "uicommon.h"
#include "Descriptors.h"
#include "zassert.h"
#include "variantptr.h"


GraphModel::GraphModel(std::shared_ptr<zeno::Graph> spGraph, QObject* parent)
    : QAbstractListModel(parent)
{
    m_graphName = QString::fromStdString(spGraph->getName());
    m_linkModel = new LinkModel(this);

    spGraph->register_addLink([&](zeno::EdgeInfo edge) -> bool {
        QPair inPair = { QString::fromStdString(edge.outNode), QString::fromStdString(edge.outParam) };
        QPair outPair = { QString::fromStdString(edge.inNode), QString::fromStdString(edge.inParam) };
        _addLink(inPair, outPair);
        return true;
    });

    spGraph->register_removeLink([&](zeno::EdgeInfo edge) -> bool {
        return _removeLink(edge);
    });
}

GraphModel::~GraphModel()
{
    if (std::shared_ptr<zeno::Graph> coreGraph = m_spCoreGraph.lock())
    {
        bool ret = coreGraph->unregister_createNode(cbCreateNode);
        ZASSERT_EXIT(ret);
    }
}

int GraphModel::indexFromId(const QString& name) const
{
    if (m_name2Row.find(name) == m_name2Row.end())
        return -1;
    return m_name2Row[name];
}

QModelIndex GraphModel::indexFromName(const QString& name) const {
    return createIndex(indexFromId(name), 0);
}

void GraphModel::addLink(const QString& fromNodeStr, const QString& fromParamStr,
    const QString& toNodeStr, const QString& toParamStr)
{
    _addLink(qMakePair(fromNodeStr, fromParamStr), qMakePair(toNodeStr, toParamStr));
}

void GraphModel::addLink(const zeno::EdgeInfo& link)
{
    //TODO
}

QString GraphModel::name() const
{
    return m_graphName;
}

void GraphModel::removeLink(const zeno::EdgeInfo& link)
{
    std::shared_ptr<zeno::Graph> spGraph = m_spCoreGraph.lock();
    ZASSERT_EXIT(spGraph);
    spGraph->removeLink(link);
}

QVariant GraphModel::removeLink(const QString& nodeName, const QString& paramName, bool bInput)
{
    if (bInput)
    {
        ParamsModel* toParamM = m_nodes[nodeName]->params;
        QModelIndex toIndex = toParamM->paramIdx(paramName, bInput);
        int nRow = toParamM->removeLink(toIndex);
       
        if (nRow != -1)
        {
            QModelIndex linkIndex = m_linkModel->index(nRow);
            QVariant var = m_linkModel->data(linkIndex, ROLE_LINK_FROMPARAM_INFO);
            QVariantList varList = var.toList();

            ParamsModel* fromParamM = m_nodes[varList[0].toString()]->params;
            QModelIndex fromIndex = fromParamM->paramIdx(varList[1].toString(), varList[2].toBool());
            fromParamM->removeLink(fromIndex);
     
            m_linkModel->removeRows(nRow, 1);
            return var;
        }
    }
    return QVariant();
}

QString GraphModel::owner() const
{
    if (auto pItem = qobject_cast<NodeItem*>(parent()))
    {
        auto spNode = pItem->spNode.lock();
        return spNode ? QString::fromStdString(spNode->name) : "";
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
    NodeItem* item = m_nodes[m_row2name[index.row()]];

    switch (role) {
        case Qt::DisplayRole:
        case ROLE_NODE_NAME: {
            auto spNode = item->spNode.lock();
            ZASSERT_EXIT(spNode, QVariant());
            return QString::fromStdString(spNode->name);
        }
        case ROLE_CLASS_NAME: {
            auto spNode = item->spNode.lock();
            ZASSERT_EXIT(spNode, QVariant());
            return QString::fromStdString(spNode->nodecls);
        }
        case ROLE_OBJPOS:
            return QVariantList({ item->pos.x(), item->pos.y() });
        case ROLE_PARAMS:
        {
            return QVariantPtr<ParamsModel>::asVariant(item->params);
        }
        case ROLE_SUBGRAPH:
        {
            if (item->optSubgraph.has_value())
                return QVariantPtr<GraphModel>::asVariant(item->optSubgraph.value());
            else
                return QVariant();
        }
        case ROLE_GRAPH:
        {
            return QVariantPtr<GraphModel>::asVariant(const_cast<GraphModel*>(this));
        }
        case ROLE_INPUTS:
        {
            PARAMS_INFO inputs;
            //TODO
            return QVariant::fromValue(inputs);
        }
        case ROLE_NODEDATA:
        {
            zeno::NodeData data;
            //TODO
            return QVariant::fromValue(data);
        }
        case ROLE_NODETYPE:
        {
            //TODO
        }
        case ROLE_OBJPATH:
        {
            //TODO
        }
        default:
            return QVariant();
    }
}

bool GraphModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    NodeItem* item = m_nodes[m_row2name[index.row()]];

    switch (role) {
        case ROLE_CLASS_NAME: {
            //TODO: rename by core graph
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
        case ROLE_OBJPOS:
        {
            if (value.type() == QVariant::PointF) {
                item->pos = value.toPoint();
            }
            else {
                QVariantList lst = value.toList();
                if (lst.size() != 2)
                    return false;
                item->pos = QPointF{ lst[0].toFloat(), lst[1].toFloat() };
            }
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
        case ROLE_INPUTS:
        {

        }
    }
    return false;
}

QModelIndexList GraphModel::match(const QModelIndex& start, int role,
    const QVariant& value, int hits,
    Qt::MatchFlags flags) const
{
    return QModelIndexList();
}

QList<SEARCH_RESULT> GraphModel::search(const QString& content, SearchType searchType, SearchOpt searchOpts) const
{
    //TODO:
    return {};
}

GraphModel* GraphModel::getGraphByPath(const QString& objPath)
{
     QStringList items = objPath.split('/', Qt::SkipEmptyParts);
     if (items.empty())
         return this;

     QString item = items[0];
     if (m_nodes.find(item) == m_nodes.end()) {
         return nullptr;
     }

     NodeItem* pItem = m_nodes[item];
     items.removeAt(0);
     QString leftPath = items.join('/');
     if (leftPath.isEmpty()) {
         return this;
     }
     ZASSERT_EXIT(pItem->optSubgraph.has_value(), nullptr);
     return pItem->optSubgraph.value()->getGraphByPath(leftPath);
}

void GraphModel::undo()
{

}

void GraphModel::redo()
{

}

void GraphModel::beginTransaction(const QString& name)
{

}

void GraphModel::endTransaction()
{

}

void GraphModel::_addLink(QPair<QString, QString> fromParam, QPair<QString, QString> toParam)
{
    QModelIndex from, to;

    ParamsModel* fromParams = m_nodes[fromParam.first]->params;
    ParamsModel* toParams = m_nodes[toParam.first]->params;

    from = fromParams->paramIdx(fromParam.second, false);
    to = toParams->paramIdx(toParam.second, true);
    
    if (from.isValid() && to.isValid())
    {
        if (toParams->getParamlinkCount(to) > 0)
            removeLink(toParam.first, toParam.second, true);
        QModelIndex linkIdx = m_linkModel->addLink(from, to);
        fromParams->addLink(from, linkIdx);
        toParams->addLink(to, linkIdx);
    }
}

bool GraphModel::_removeLink(const zeno::EdgeInfo& edge)
{
    QString outNode = QString::fromStdString(edge.outNode);
    QString inNode = QString::fromStdString(edge.inNode);
    QString outParam = QString::fromStdString(edge.outParam);
    QString inParam = QString::fromStdString(edge.inParam);

    QModelIndex from, to;

    ParamsModel* fromParams = m_nodes[outNode]->params;
    ParamsModel* toParams = m_nodes[inNode]->params;

    from = fromParams->paramIdx(outParam, false);
    to = toParams->paramIdx(inParam, true);
    if (from.isValid() && to.isValid())
    {
        QModelIndex linkIdx = fromParams->removeOneLink(from, edge);
        QModelIndex linkIdx2 = toParams->removeOneLink(to, edge);
        ZASSERT_EXIT(linkIdx == linkIdx2, false);
        m_linkModel->removeRow(linkIdx.row());
    }
    return true;
}

void GraphModel::registerCoreNotify(std::shared_ptr<zeno::Graph> coreGraph)
{
    m_spCoreGraph = coreGraph;
    cbCreateNode = coreGraph->register_createNode([&](const std::string& name, std::weak_ptr<zeno::INode> spNode) {
        auto coreNode = spNode.lock();
        _appendNode(coreNode);
    });
}

zeno::NodeData GraphModel::createNode(const QString& nodeCls, const QPointF& pos)
{
    zeno::NodeData node;
    std::shared_ptr<zeno::Graph> spGraph = m_spCoreGraph.lock();
    if (!spGraph)
        return node;

    spGraph->createNode(nodeCls.toStdString());
    return node;
}

void GraphModel::_appendNode(std::shared_ptr<zeno::INode> spNode)
{
    ZASSERT_EXIT(spNode);

    int nRows = m_nodes.size();

    beginInsertRows(QModelIndex(), nRows, nRows);

    NodeItem* pItem = new NodeItem(this);
    pItem->setParent(this);
    pItem->spNode = spNode;

    pItem->params = new ParamsModel(spNode, pItem);

    const QString& name = QString::fromStdString(spNode->name);
    m_row2name[nRows] = name;
    m_name2Row[name] = nRows;
    m_nodes.insert(name, pItem);

    endInsertRows();

    pItem->params->setNodeIdx(createIndex(nRows, 0));
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

void GraphModel::removeNode(QString ident)
{
    int row = m_name2Row[ident];
    //core remove
    removeRow(row);
}

QHash<int, QByteArray> GraphModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[ROLE_CLASS_NAME] = "classname";
    roles[ROLE_NODE_NAME] = "name";
    roles[ROLE_PARAMS] = "params";
    roles[ROLE_LINKS] = "linkModel";
    roles[ROLE_OBJPOS] = "pos";
    roles[ROLE_SUBGRAPH] = "subgraph";
    return roles;
}

bool GraphModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row);

    QString id = m_row2name[row];
    NodeItem* pItem = m_nodes[id];
    //pItem->params->clear();

    for (int r = row + 1; r < rowCount(); r++)
    {
        const QString& id_ = m_row2name[r];
        m_row2name[r - 1] = id_;
        m_name2Row[id_] = r - 1;
    }

    m_row2name.remove(rowCount() - 1);
    m_name2Row.remove(id);
    m_nodes.remove(id);

    delete pItem;

    endRemoveRows();
    return true;
}


void GraphModel::updateParamName(QModelIndex nodeIdx, int row, QString newName)
{
    NodeItem* item = m_nodes[m_row2name[nodeIdx.row()]];
    QModelIndex paramIdx = item->params->index(row, 0);
    item->params->setData(paramIdx, newName, ROLE_PARAM_NAME);
}

void GraphModel::removeParam(QModelIndex nodeIdx, int row)
{
    NodeItem* item = m_nodes[m_row2name[nodeIdx.row()]];
    item->params->removeRow(row);
}

void GraphModel::removeLink(int row)
{
    m_linkModel->removeRow(row);
}

ParamsModel* GraphModel::params(QModelIndex nodeIdx)
{
    NodeItem* item = m_nodes[m_row2name[nodeIdx.row()]];
    return item->params;
}

GraphModel* GraphModel::subgraph(QModelIndex nodeIdx) {
    NodeItem* item = m_nodes[m_row2name[nodeIdx.row()]];
    if (item->optSubgraph.has_value()) {
        return item->optSubgraph.value();
    }
    return nullptr;
}

QModelIndex GraphModel::nodeIdx(const QString& ident) const
{
    int nRow = m_name2Row[ident];
    return createIndex(nRow, 0);
}
