#ifndef __TRANSCATION_COMMAND_H__
#define __TRANSCATION_COMMAND_H__

#include <QUndoCommand>
#include "modeldata.h"
//#include "subgraphmodel.h"

class SubGraphModel;

class AddNodeCommand : public QUndoCommand
{
public:
    AddNodeCommand(int row, const QString& id, const NODE_DATA& data, SubGraphModel *pModel);
    void redo();
    void undo();

private:
    int m_row;
    QString m_id;
    SubGraphModel* m_model;
    NODE_DATA m_data;
};

class RemoveNodeCommand : public QUndoCommand
{
public:
    RemoveNodeCommand(int row, const NODE_DATA& data, SubGraphModel* pModel);
    void redo();
    void undo();

private:
    NODE_DATA m_data;
    int m_row;
    SubGraphModel* m_model;
};

class AddRemoveLinkCommand : public QUndoCommand
{
public:
    AddRemoveLinkCommand(EdgeInfo info, bool bAdded, SubGraphModel *pModel);
    void redo();
    void undo();

private:
    EdgeInfo m_info;
    SubGraphModel* m_model;
    bool m_bAdded;
};

#endif