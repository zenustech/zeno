#ifndef __NODE_ITEM_H__
#define __NODE_ITEM_H__

#include <zenomodel/include/modeldata.h>
#include <QString>
#include <QPoint>
#include <QObject>

class PanelParamModel;
class NodeParamModel;
class IGraphsModel;
class TreeNodeItem;

//base QObject?
class NodeItem : public QObject
{
    Q_OBJECT
public:
    NodeItem(QObject *parent = nullptr);
    QModelIndex nodeIdx() const;

public:
    QString objid;
    QString objCls;
    QString customName;
    NODE_TYPE type;
    QPointF viewpos;
    int options;
    PARAMS_INFO paramNotDesc;

    PanelParamModel* panelParams;
    NodeParamModel* nodeParams;
    TreeNodeItem *treeItem;     //when apply tree layout

    bool bCollasped;
};

struct TreeNodeItem : public QStandardItem
{
    TreeNodeItem(const NODE_DATA& nodeData, IGraphsModel* pGraphsModel);
    ~TreeNodeItem();

    QVariant data(int role = Qt::UserRole + 1) const override;
    void setData(const QVariant &value, int role) override;
    int id2Row(const QString& ident) const {
        if (m_ident2row.find(ident) == m_ident2row.end())
            return -1;
        return m_ident2row[ident];
    }
    void addNode(const NODE_DATA& nodeData, IGraphsModel* pModel);
    void appendRow(TreeNodeItem* pChildItem);
    void removeNode(const QString& ident, IGraphsModel* pModel);
    NODE_DATA expData() const;
    QString objClass() const;
    QString objName() const;
    QModelIndex childIndex(const QString& ident) const;
    TreeNodeItem* childItem(const QString& ident);

private:
    bool checkCustomName(const QString& name);

    NodeItem* m_item;
    QHash<QString, int> m_ident2row;
};

#endif