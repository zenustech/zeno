#ifndef __TRANSCATION_COMMAND_H__
#define __TRANSCATION_COMMAND_H__

#include <QUndoCommand>
#include <zenoui/model/modeldata.h>

class SubGraphModel;
class GraphsModel;

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

class AddLinkCommand : public QUndoCommand
{
public:
    AddLinkCommand(EdgeInfo info, GraphsModel* pModel, QPersistentModelIndex subgIdx);
	void redo() override;
	void undo() override;

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
    void redo() override;
    void undo() override;

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
    UpdateDataCommand(const QString& nodeid, const PARAM_UPDATE_INFO& updateInfo, GraphsModel* pModel, QPersistentModelIndex subgIdx);
    void redo() override;
    void undo() override;

private:
    PARAM_UPDATE_INFO m_updateInfo;
    QString m_nodeid;
    QPersistentModelIndex m_subgIdx;
    GraphsModel* m_model;
};

class UpdateSockDeflCommand : public QUndoCommand
{
public:
    UpdateSockDeflCommand(const QString& nodeid, const PARAM_UPDATE_INFO& updateInfo, GraphsModel* pModel, QPersistentModelIndex subgIdx);
    void redo() override;
    void undo() override;

private:
    PARAM_UPDATE_INFO m_updateInfo;
    QString m_nodeid;
    QPersistentModelIndex m_subgIdx;
    GraphsModel* m_model;
};

class UpdateStateCommand : public QUndoCommand
{
public:
    UpdateStateCommand(const QString& nodeid, STATUS_UPDATE_INFO info, GraphsModel* pModel, QPersistentModelIndex subgIdx);
    void redo() override;
    void undo() override;

private:
    QString m_nodeid;
    STATUS_UPDATE_INFO m_info;
    QPersistentModelIndex m_subgIdx;
    GraphsModel* m_pModel;
};

class UpdateSocketCommand : public QUndoCommand
{
public:
    UpdateSocketCommand(const QString& nodeid, SOCKET_UPDATE_INFO info, GraphsModel* pModel, QPersistentModelIndex subgIdx);
    void redo() override;
    void undo() override;

private:
    SOCKET_UPDATE_INFO m_info;
    QString m_nodeid;
    QPersistentModelIndex m_subgIdx;
    GraphsModel* m_pModel;
};


#endif
