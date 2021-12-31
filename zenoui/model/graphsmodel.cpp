#include "subgraphmodel.h"
#include "graphsmodel.h"
#include "modelrole.h"
#include "../util/uihelper.h"


GraphsModel::GraphsModel(QObject *parent)
    : QStandardItemModel(parent)
    , m_selection(nullptr)
{
    m_selection = new QItemSelectionModel(this);
}

GraphsModel::~GraphsModel()
{
}

QItemSelectionModel* GraphsModel::selectionModel() const
{
    return m_selection;
}

void GraphsModel::setFilePath(const QString& fn)
{
    m_filePath = fn;
}

SubGraphModel* GraphsModel::subGraph(const QString& name)
{
    const QModelIndexList& lst = this->match(index(0, 0), ROLE_OBJNAME, name, 1, Qt::MatchExactly);
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

SubGraphModel* GraphsModel::currentGraph()
{
    return subGraph(m_selection->currentIndex().row());
}

void GraphsModel::switchSubGraph(const QString& graphName)
{
    QModelIndex startIndex = createIndex(0, 0, nullptr);
    const QModelIndexList &lst = this->match(startIndex, ROLE_OBJNAME, graphName, 1, Qt::MatchExactly);
    if (lst.size() == 1)
    {
        m_selection->setCurrentIndex(lst[0], QItemSelectionModel::Current);
    }
}

void GraphsModel::newSubgraph(const QString &graphName)
{
    QModelIndex startIndex = createIndex(0, 0, nullptr);
    const QModelIndexList &lst = this->match(startIndex, ROLE_OBJNAME, graphName, 1, Qt::MatchExactly);
    if (lst.size() == 1)
    {
        m_selection->setCurrentIndex(lst[0], QItemSelectionModel::Current);
    }
    else
    {
        SubGraphModel *subGraphModel = new SubGraphModel(this);
        subGraphModel->setName(graphName);
        appendSubGraph(subGraphModel);
        m_selection->setCurrentIndex(index(rowCount() - 1, 0), QItemSelectionModel::Current);
    }
}

void GraphsModel::reloadSubGraph(const QString& graphName)
{
    initDescriptors();
    SubGraphModel *pReloadModel = subGraph(graphName);
    Q_ASSERT(pReloadModel);
    NODES_DATA datas = pReloadModel->nodes();
    pReloadModel->clear();
    pReloadModel->blockSignals(true);
    for (auto data : datas)
    {
        pReloadModel->appendItem(data);
    }
    pReloadModel->blockSignals(false);
    pReloadModel->reload();
}

int GraphsModel::graphCounts() const
{
    return rowCount();
}

void GraphsModel::initDescriptors()
{
    NODE_DESCS descs = UiHelper::loadDescsFromTempFile();
    NODE_DESCS subgDescs = getSubgraphDescs();
    descs.insert(subgDescs);

    NODE_DESC blackBoard;
    blackBoard.categories.push_back("layout");
    descs["Blackboard"] = blackBoard;

    setDescriptors(descs);
}

NODE_DESCS GraphsModel::getSubgraphDescs()
{
    NODE_DESCS descs;
    for (int r = 0; r < this->rowCount(); r++)
    {
        QModelIndex index = this->index(r, 0);
        Q_ASSERT(index.isValid());
        SubGraphModel *pModel = static_cast<SubGraphModel *>(index.data(ROLE_GRAPHPTR).value<void *>());
        const QString& graphName = pModel->name();
        if (graphName == "main")
            continue;

        QString subcategory = "subgraph";
        INPUT_SOCKETS subInputs;
        OUTPUT_SOCKETS subOutputs;
        for (int i = 0; i < pModel->rowCount(); i++)
        {
            QModelIndex idx = pModel->index(i, 0);
            const QString& nodeName = idx.data(ROLE_OBJNAME).toString();
            PARAMS_INFO params = idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
            if (nodeName == "SubInput")
            {
                QString n_type = params["type"].value.toString();
                QString n_name = params["name"].value.toString();
                auto n_defl = params["defl"].value;

                SOCKET_INFO info;
                info.name = n_name;
                info.type = n_type;
                info.defaultValue = n_defl; 

                INPUT_SOCKET inputSock;
                inputSock.info = info;
                subInputs.insert(n_name, inputSock);
            }
            else if (nodeName == "SubOutput")
            {
                auto n_type = params["type"].value.toString();
                auto n_name = params["name"].value.toString();
                auto n_defl = params["defl"].value;

                SOCKET_INFO info;
                info.name = n_name;
                info.type = n_type;
                info.defaultValue = n_defl; 

                OUTPUT_SOCKET outputSock;
                outputSock.info = info;
                subOutputs.insert(n_name, outputSock);
            }
            else if (nodeName == "SubCategory")
            {
                subcategory = params["name"].value.toString();
            }
        }

        subInputs.insert(m_nodesDesc["Subgraph"].inputs);
        subOutputs.insert(m_nodesDesc["Subgraph"].outputs);
        
        NODE_DESC desc;
        desc.inputs = subInputs;
        desc.outputs = subOutputs;
        desc.categories.push_back(subcategory);
        desc.is_subgraph = true;
        descs[graphName] = desc;
    }
    return descs;
}

NODE_DESCS GraphsModel::descriptors() const
{
    return m_nodesDesc;
}

void GraphsModel::setDescriptors(const NODE_DESCS& nodeDescs)
{
    m_nodesDesc = nodeDescs;
    m_nodesCate.clear();
    for (auto it = m_nodesDesc.constBegin(); it != m_nodesDesc.constEnd(); it++)
    {
        const QString& name = it.key();
        const NODE_DESC& desc = it.value();
        for (auto cate : desc.categories)
        {
            m_nodesCate[cate].name = cate;
            m_nodesCate[cate].nodes.push_back(name);
        }
    }
}

void GraphsModel::appendSubGraph(SubGraphModel* pGraph)
{
    QStandardItem *pItem = new QStandardItem;
    QString graphName = pGraph->name();
    QVariant var(QVariant::fromValue(static_cast<void *>(pGraph)));
    pItem->setText(graphName);
    pItem->setData(var, ROLE_GRAPHPTR);
    pItem->setData(graphName, ROLE_OBJNAME);
    appendRow(pItem);
}

void GraphsModel::removeGraph(int idx)
{
    removeRow(idx);
}

NODE_CATES GraphsModel::getCates()
{
    return m_nodesCate;
}

void GraphsModel::onCurrentIndexChanged(int row)
{
    const QString& graphName = data(index(row, 0), ROLE_OBJNAME).toString();
    switchSubGraph(graphName);
}

void GraphsModel::onRemoveCurrentItem()
{
    removeGraph(m_selection->currentIndex().row());
    //switch to main.
    QModelIndex startIndex = createIndex(0, 0, nullptr);
    //to ask: if not main scene?
    const QModelIndexList &lst = this->match(startIndex, ROLE_OBJNAME, "main", 1, Qt::MatchExactly);
    if (lst.size() == 1)
        m_selection->setCurrentIndex(index(lst[0].row(), 0), QItemSelectionModel::Current);
}
