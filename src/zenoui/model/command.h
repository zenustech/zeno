#ifndef __TRANSCATION_COMMAND_H__
#define __TRANSCATION_COMMAND_H__

#include <QUndoCommand>
#include "modeldata.h"

class SubGraphModel;
class GraphsModel;

class AddNodeCommand : public QUndoCommand
{
public:
    AddNodeCommand(const QString& id, const NODE_DATA& data, GraphsModel* pModel, QPersistentModelIndex subgIdx);
    ~AddNodeCommand();
    void redo();
    void undo();

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
    void redo();
    void undo();

private:
    QString m_id;
    NODE_DATA m_data;
    QPersistentModelIndex m_subgIdx;
    GraphsModel* m_model;
    int m_row;
};

class AddLinkCommand : public QUndoCommand
{
public:
    AddLinkCommand(EdgeInfo info, GraphsModel* pModel, QPersistentModelIndex subgIdx);
	void redo();
	void undo();

private:
	EdgeInfo m_info;
	GraphsModel* m_model;
	QPersistentModelIndex m_subgIdx;
	QPersistentModelIndex m_linkIdx;
};

class RemoveLinkCommand : public QUndoCommand
{
public:
    RemoveLinkCommand(QPersistentModelIndex linkIdx, GraphsModel* pModel, QPersistentModelIndex subgIdx);
    void redo();
    void undo();

private:
    EdgeInfo m_info;
    GraphsModel* m_model;
    QPersistentModelIndex m_subgIdx;
    QPersistentModelIndex m_linkIdx;
    bool m_bAdded;
};

class UpdateDataCommand : public QUndoCommand
{
public:
    UpdateDataCommand(const QString& nodeid, const QString& paramName, const QVariant& newValue, SubGraphModel* pModel);
    void redo();
    void undo();

private:
    QVariant m_newValue;
    QVariant m_oldValue;
    QString m_name;
    QString m_nodeid;
    SubGraphModel *m_model;
};

class UpdateStateCommand : public QUndoCommand
{
public:
    UpdateStateCommand(const QString& nodeid, int role, const QVariant& val, SubGraphModel* pModel);
    void redo();
    void undo();

private:
    QString m_nodeid;
    QVariant m_value;
    QVariant m_oldValue;
    int m_role;
    SubGraphModel* m_pModel;
};

#endif