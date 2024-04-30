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

#endif
