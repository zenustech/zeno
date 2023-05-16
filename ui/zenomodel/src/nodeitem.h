#ifndef __NODE_ITEM_H__
#define __NODE_ITEM_H__

#include "modeldata.h"
#include <QString>
#include <QPoint>
#include <QObject>

class PanelParamModel;
class NodeParamModel;

struct NodeItem
{
    QString objid;
    QString objCls;
    QString customName;
    NODE_TYPE type;
    QPointF viewpos;
    int options;
    PARAMS_INFO paramNotDesc;

    PanelParamModel* panelParams;
    NodeParamModel* nodeParams;

    bool bCollasped;

    NodeItem()
        : options(0)
        , bCollasped(false)
        , type(NORMAL_NODE)
        , panelParams(nullptr)
        , nodeParams(nullptr)
    {
    }
};

struct TreeNodeItem : public QStandardItem
{
    TreeNodeItem(const TreeNodeItem &);
    ~TreeNodeItem();

    QVariant data(int role = Qt::UserRole + 1) const override;
    void setData(const QVariant &value, int role) override;

private:
    static NODE_DATA item2NodeData(const NodeItem& item);
    bool checkCustomName(const QString& name);

    NodeItem m_item;
};

#endif