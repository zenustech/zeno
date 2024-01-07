#include "GraphModel.h"
#include "../common.h"
#include "Descriptors.h"


GraphModel::GraphModel(const QString& graphName, QObject* parent)
    : QAbstractListModel(parent)
    , m_graphName(graphName)
{
    m_linkModel = new LinkModel(this);
}

GraphModel::~GraphModel()
{
}

int GraphModel::indexFromId(const QString& ident) const
{
    return m_id2Row[ident];
}

void GraphModel::addLink(const QString& fromNodeStr, const QString& fromParamStr,
    const QString& toNodeStr, const QString& toParamStr)
{
    addLink(qMakePair(fromNodeStr, fromParamStr), qMakePair(toNodeStr, toParamStr));
}

QVariant GraphModel::removeLink(const QString& nodeIdent, const QString& paramName, bool bInput)
{
    if (bInput)
    {
        ParamsModel* toParamM = m_nodes[nodeIdent]->params;
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
        return pItem->ident;
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
    NodeItem* item = m_nodes[m_row2id[index.row()]];

    switch (role) {
    case Qt::DisplayRole:   return item->ident;
    case ROLE_OBJID:    return item->ident;
    case ROLE_OBJNAME:  return item->name;
    case ROLE_OBJPOS:   return QVariantList({ item->pos.x(), item->pos.y() });
    case ROLE_PARAMS:
    {
        return QVariant::fromValue(item->params);
    }
    case ROLE_SUBGRAPH:
    {
        if (item->pSubgraph)
            return QVariant::fromValue(item->pSubgraph);
        else
            return QVariant();
    }
    default:
        return QVariant();
    }
}

bool GraphModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    NodeItem* item = m_nodes[m_row2id[index.row()]];

    switch (role) {
        case ROLE_OBJNAME: {
            item->name = value.toString();
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
        case ROLE_OBJPOS:
        {
            QVariantList lst = value.toList();
            item->pos = QPointF{ lst[0].toFloat(), lst[1].toFloat() };
            emit dataChanged(index, index, QVector<int>{role});
            return true;
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

void GraphModel::addLink(QPair<QString, QString> fromParam, QPair<QString, QString> toParam)
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

void GraphModel::appendNode(QString ident, QString name, const QPointF& pos)
{
    auto* pDescs = Descriptors::instance();
    NODE_DESCRIPTOR desc = pDescs->getDescriptor(name);

    int nRows = m_nodes.size();

    beginInsertRows(QModelIndex(), nRows, nRows);

    NodeItem* pItem = new NodeItem(this);
    pItem->setParent(this);
    pItem->ident = ident;
    pItem->name = name;
    pItem->pos = pos;
    pItem->params = new ParamsModel(desc);

    m_row2id[nRows] = ident;
    m_id2Row[ident] = nRows;
    m_nodes.insert(ident, pItem);

    endInsertRows();

    pItem->params->setNodeIdx(createIndex(nRows, 0));
}

void GraphModel::appendSubgraphNode(QString ident, QString name, NODE_DESCRIPTOR desc, GraphModel* subgraph, const QPointF& pos)
{
    int nRows = m_nodes.size();
    beginInsertRows(QModelIndex(), nRows, nRows);

    NodeItem* pItem = new NodeItem(this);
    pItem->setParent(this);
    pItem->ident = ident;
    pItem->name = name;
    pItem->pos = pos;
    pItem->params = new ParamsModel(desc);
    pItem->pSubgraph = subgraph;
    subgraph->setParent(pItem);

    m_row2id[nRows] = ident;
    m_id2Row[ident] = nRows;
    m_nodes.insert(ident, pItem);

    endInsertRows();
    pItem->params->setNodeIdx(createIndex(nRows, 0));
}

void GraphModel::removeNode(QString ident)
{
    int row = m_id2Row[ident];
    removeRow(row);
}

QHash<int, QByteArray> GraphModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[ROLE_OBJNAME] = "name";
    roles[ROLE_OBJID] = "ident";
    roles[ROLE_PARAMS] = "params";
    roles[ROLE_LINKS] = "linkModel";
    roles[ROLE_OBJPOS] = "pos";
    roles[ROLE_SUBGRAPH] = "subgraph";
    return roles;
}

bool GraphModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row);

    QString id = m_row2id[row];
    NodeItem* pItem = m_nodes[id];
    //pItem->params->clear();

    for (int r = row + 1; r < rowCount(); r++)
    {
        const QString& id_ = m_row2id[r];
        m_row2id[r - 1] = id_;
        m_id2Row[id_] = r - 1;
    }

    m_row2id.remove(rowCount() - 1);
    m_id2Row.remove(id);
    m_nodes.remove(id);

    delete pItem;

    endRemoveRows();
    return true;
}


void GraphModel::updateParamName(QModelIndex nodeIdx, int row, QString newName)
{
    NodeItem* item = m_nodes[m_row2id[nodeIdx.row()]];
    QModelIndex paramIdx = item->params->index(row, 0);
    item->params->setData(paramIdx, newName, ROLE_OBJNAME);
}

void GraphModel::removeParam(QModelIndex nodeIdx, int row)
{
    NodeItem* item = m_nodes[m_row2id[nodeIdx.row()]];
    item->params->removeRow(row);
}

void GraphModel::removeLink(int row)
{
    m_linkModel->removeRow(row);
}

ParamsModel* GraphModel::params(QModelIndex nodeIdx)
{
    NodeItem* item = m_nodes[m_row2id[nodeIdx.row()]];
    return item->params;
}

GraphModel* GraphModel::subgraph(QModelIndex nodeIdx) {
    NodeItem* item = m_nodes[m_row2id[nodeIdx.row()]];
    return item->pSubgraph;
}

QModelIndex GraphModel::nodeIdx(const QString& ident) const
{
    int nRow = m_id2Row[ident];
    return createIndex(nRow, 0);
}
