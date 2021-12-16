#include "subgraphmodel.h"
#include "graphsmodel.h"
#include "modelrole.h"


GraphsModel::GraphsModel(QObject *parent)
    : QStandardItemModel(parent)
    , m_selection(nullptr)
    , m_currentIndex(0)
{
    m_selection = new QItemSelectionModel(this);
    connect(m_selection, SIGNAL(selectionChanged(const QItemSelection&, const QItemSelection&)),
            this, SLOT(onSelectionChanged(const QItemSelection&, const QItemSelection&)));
}

GraphsModel::~GraphsModel()
{
}

SubGraphModel* GraphsModel::subGraph(const QString& name)
{
    const QModelIndexList& lst = this->match(QModelIndex(), ROLE_OBJNAME, name, 1, Qt::MatchExactly);
    if (lst.size() > 0)
    {
        SubGraphModel* pModel = static_cast<SubGraphModel*>(lst[0].data(ROLE_GRAPHPTR).value<void*>());
        return pModel;
    }
    return nullptr;
}

SubGraphModel* GraphsModel::subGraph(int idx)
{
    QModelIndex index = this->index(idx, 0);
    if (index.isValid())
    {
        SubGraphModel *pModel = static_cast<SubGraphModel *>(index.data(ROLE_GRAPHPTR).value<void *>());
        return pModel;
    }
    return nullptr;
}

int GraphsModel::graphCounts() const
{
    return this->rowCount();
}

NODES_PARAMS GraphsModel::getNodesTemplate() const
{
    return m_nodesDict;
}

void GraphsModel::setNodesTemplate(const NODES_PARAMS &nodesParams)
{
    m_nodesDict = nodesParams;
}

void GraphsModel::onCurrentIndexChanged(int row)
{
    if (m_currentIndex == row)
        return;

    m_currentIndex = row;
    emit itemSelected(m_currentIndex);
    //m_selection->select(index(row, 0), QItemSelectionModel::Select);
}

void GraphsModel::onSelectionChanged(const QItemSelection& selected, const QItemSelection& deselected)
{
    QModelIndexList lst = selected.indexes();
    if (!lst.isEmpty())
    {
        QModelIndex idx = lst.at(0);
        emit itemSelected(idx.row());
    }
    lst = deselected.indexes();
    if (!lst.isEmpty())
    {
        QModelIndex idx = lst.at(0);
    }
}