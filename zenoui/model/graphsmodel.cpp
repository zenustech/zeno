#include "subgraphmodel.h"
#include "graphsmodel.h"
#include "modelrole.h"
#include "../util/uihelper.h"


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
    //todo: setdefault cat
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