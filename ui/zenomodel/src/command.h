#ifndef __TRANSCATION_COMMAND_H__
#define __TRANSCATION_COMMAND_H__

#include <QUndoCommand>
#include "modeldata.h"

class GraphsModel;
class IGraphsModel;

class AddNodeCommand : public QUndoCommand
{
public:
    AddNodeCommand(const QString& id, const NODE_DATA& data, GraphsModel* pModel, QPersistentModelIndex subgIdx);
    ~AddNodeCommand();
    void redo() override;
    void undo() override;

private:
    QString m_id;
    NODE_DATA m_data;
    QPersistentModelIndex m_subgIdx;
    GraphsModel* m_model;
};

class RemoveNodeCommand : public QUndoCommand
{
public:
    RemoveNodeCommand(int row, NODE_DATA data, GraphsModel* pModel, QPersistentModelIndex subgIdx);
    ~RemoveNodeCommand();
    void redo() override;
    void undo() override;

private:
    QString m_id;
    NODE_DATA m_data;
    QPersistentModelIndex m_subgIdx;
    GraphsModel* m_model;
    int m_row;
};

class LinkCommand : public QUndoCommand
{
public:
    LinkCommand(bool bAddLink, const EdgeInfo& link, GraphsModel *pModel);
    void redo() override;
    void undo() override;

private:
    const bool m_bAdd;
    const EdgeInfo m_link;
    GraphsModel* m_model;
};

class UpdateBlackboardCommand : public QUndoCommand
{
public:
    UpdateBlackboardCommand(const QString &nodeid, BLACKBOARD_INFO newInfo, BLACKBOARD_INFO oldInfo,
                            GraphsModel *pModel, QPersistentModelIndex subgIdx);
    void redo() override;
    void undo() override;

private:
    BLACKBOARD_INFO m_oldInfo;
    BLACKBOARD_INFO m_newInfo;
    QString m_nodeid;
    QPersistentModelIndex m_subgIdx;
    GraphsModel *m_pModel;
};

class ImportNodesCommand : public QUndoCommand
{
public:
    ImportNodesCommand(const QMap<QString, NODE_DATA>& nodes, const QList<EdgeInfo>& links, QPointF pos, GraphsModel *pModel, QPersistentModelIndex subgIdx);
    void redo() override;
    void undo() override;

private:
    const QMap<QString, NODE_DATA> m_nodes;
    const QList<EdgeInfo> m_links;
    QPersistentModelIndex m_subgIdx;
    GraphsModel* m_model;
    QPointF m_pos;
};

class ModelDataCommand : public QUndoCommand
{
public:
    ModelDataCommand(IGraphsModel* pModel, const QModelIndex& idx, const QVariant& oldData, const QVariant& newData, int role);
    void redo() override;
    void undo() override;

private:
    void ensureIdxValid();

    IGraphsModel* m_model;
    const QVariant m_oldData;
    const QVariant m_newData;
    QString m_objPath;
    QPersistentModelIndex m_index;
    const int m_role;
};

class UpdateSubgDescCommand : public QUndoCommand
{
public:
    UpdateSubgDescCommand(IGraphsModel* pModel, const QString& subgraphName, const NODE_DESC newDesc);
    void redo() override;
    void undo() override;

private:
    IGraphsModel* m_model;
    const QString m_subgName;
    NODE_DESC m_oldDesc;
    NODE_DESC m_newDesc;
};


#endif
