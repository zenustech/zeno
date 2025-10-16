#ifndef __TRANSCATION_COMMAND_H__
#define __TRANSCATION_COMMAND_H__

#include <QUndoCommand>
#include <zeno/core/data.h>
#include "GraphModel.h"
#include "model/graphsmanager.h"

class AddNodeCommand : public QUndoCommand
{
public:
    AddNodeCommand(const QString& cate, zeno::NodeData& nodedata, const QStringList& graphPath);
    ~AddNodeCommand();
    void redo() override;
    void undo() override;
    zeno::NodeData getNodeData();

private:
    GraphModel* m_model;
    QStringList m_graphPath;
    zeno::NodeData m_nodeData;
    std::pair<float, float> m_pos;
    QString m_cate;
};

class RemoveNodeCommand : public QUndoCommand
{
public:
    RemoveNodeCommand(zeno::NodeData& nodeData, const QStringList& graphPath);
    ~RemoveNodeCommand();
    void redo() override;
    void undo() override;

private:
    GraphModel* m_model;
    QStringList m_graphPath;
    zeno::NodeData m_nodeData;
    QString m_cate;
};

class LinkCommand : public QUndoCommand
{
public:
    LinkCommand(bool bAddLink, const zeno::EdgeInfo& link, const QStringList& graphPath);
    void redo() override;
    void undo() override;

private:
    const bool m_bAdd;
    const zeno::EdgeInfo m_link;

    GraphModel* m_model;
    QStringList m_graphPath;

    QString m_lastViewNodeName;
};

class ModelDataCommand : public QUndoCommand
{
public:
    ModelDataCommand(const QModelIndex& index, const QVariant& oldData, const QVariant& newData, int role, const QStringList& graphPath);
    void redo() override;
    void undo() override;

private:
    const QVariant m_oldData;
    const QVariant m_newData;
    const int m_role;
    QStringList m_graphPath;

    GraphModel* m_model;
    QString m_nodeName;
    QString m_paramName;
};

class NodeStatusCommand : public QUndoCommand
{
public:
    NodeStatusCommand(zeno::NodeStatus status, const QString& name, bool bOn, const QStringList& graphPath);
    void redo() override;
    void undo() override;

private:
    zeno::NodeStatus m_status;
    QString m_lastViewNodeName;

    bool m_bOn;
    QStringList m_graphPath;

    GraphModel* m_model;
    QString m_nodeName;
};

#endif
