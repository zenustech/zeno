#ifndef __TRANSCATION_COMMAND_H__
#define __TRANSCATION_COMMAND_H__

#include <QUndoCommand>
#include <zeno/core/data.h>
#include "GraphModel.h"
#include "model/graphsmanager.h"

class AddNodeCommand : public QUndoCommand
{
public:
    AddNodeCommand(const QString& cate, zeno::NodeData& nodedata, QStringList& graphPath);
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
    RemoveNodeCommand(zeno::NodeData& nodeData, QStringList& graphPath);
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
    LinkCommand(bool bAddLink, const zeno::EdgeInfo& link, QStringList& graphPath);
    void redo() override;
    void undo() override;

private:
    const bool m_bAdd;
    const zeno::EdgeInfo m_link;
    GraphModel* m_model;
    QStringList m_graphPath;
};

class ModelDataCommand : public QUndoCommand
{
public:
    ModelDataCommand(const QModelIndex& index, const QVariant& oldData, const QVariant& newData, int role, QStringList& graphPath);
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
    NodeStatusCommand(bool isSetView, const QString& name, bool bOn, QStringList& graphPath);
    void redo() override;
    void undo() override;

private:
    bool m_isSetView;
    bool m_On;
    QStringList m_graphPath;

    GraphModel* m_model;
    QString m_nodeName;
};

#endif
