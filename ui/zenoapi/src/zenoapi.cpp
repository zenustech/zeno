#include "zenoapi.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/nodesmgr.h>


ZENO_ERROR Zeno_NewFile()
{
    IGraphsModel* pModel = GraphsManagment::instance().newFile();
    return pModel != nullptr;
}

ZENO_HANDLE Zeno_CreateGraph(const std::string& name)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;

    const QString& qsName = QString::fromStdString(name);
    pModel->newSubgraph(QString::fromStdString(name));
    QModelIndex subgIdx = pModel->index(qsName);
    return subgIdx.internalId();
}

ZENO_ERROR Zeno_DeleteGraph(ZENO_HANDLE hSubgraph)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;

    QModelIndex subgIdx = pModel->subgIndex(hSubgraph);
    pModel->removeSubGraph(subgIdx.data(ROLE_OBJNAME).toString());
    return 0;
}

ZENO_HANDLE Zeno_GetGraph(const std::string &name)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;

    QModelIndex subgIdx = pModel->index(QString::fromStdString(name));
    return subgIdx.internalId();
}

ZENO_ERROR Zeno_RenameGraph(ZENO_HANDLE hSubgraph, const std::string& newName)
{
    return 0;
}

int Zeno_GetCount()
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;
    return pModel->rowCount();
}

ZENO_HANDLE Zeno_GetItem(int idx)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;

    QModelIndex subgIdx = pModel->index(idx, 0);
    return subgIdx.internalId();
}

ZENO_HANDLE Zeno_AddNode(ZENO_HANDLE hGraph, const std::string& nodeCls)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;

    QModelIndex subgIdx = pModel->subgIndex(hGraph);
    if (!subgIdx.isValid())
        return -1;

    QString ident = NodesMgr::createNewNode(pModel, subgIdx, QString::fromStdString(nodeCls), QPointF(0, 0));
    return pModel->index(ident, subgIdx).internalId();
}

ZENO_HANDLE Zeno_GetNode(const std::string& ident)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;

    return pModel->nodeIndex(QString::fromStdString(ident)).internalId();
}

ZENO_ERROR Zeno_DeleteNode(ZENO_HANDLE hNode)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;

    QModelIndex subgIdx = pModel->subgByNodeId(hNode);
    QModelIndex nodeIdx = pModel->nodeIndex(hNode);
    pModel->removeNode(nodeIdx.data(ROLE_OBJID).toString(), subgIdx);
    return 0;
}

ZENO_ERROR Zeno_GetName(ZENO_HANDLE hNode, std::string& name)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;
    QModelIndex idx = pModel->nodeIndex(hNode);
    name = idx.data(ROLE_OBJNAME).toString().toStdString();
    return 0;
}

ZENO_ERROR Zeno_GetIdent(ZENO_HANDLE hNode, std::string& ident)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return -1;
    QModelIndex idx = pModel->nodeIndex(hNode);
    ident = idx.data(ROLE_OBJID).toString().toStdString();
    return 0;
}

//io
ZENO_ERROR Zeno_OpenFile(const std::string &fn)
{
    return 0;
}

ZENO_ERROR Zeno_saveFile(const std::string &fn)
{
    return 0;
}
