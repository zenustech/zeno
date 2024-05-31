#ifndef __ZENO_NODE_BASE_H__
#define __ZENO_NODE_BASE_H__

#include <QtWidgets>
#include "control/renderparam.h"
#include "zenosvgitem.h"
#include "zenobackgrounditem.h"
#include "nodesys_common.h"
#include "nodeeditor/gv/zenosocketitem.h"
#include "nodeeditor/gv/zenoparamwidget.h"
#include <zeno/core/data.h>
#include "nodeeditor/gv/zgraphicslayout.h"
#include "nodeeditor/gv/zgraphicslayoutitem.h"
#include "nodeeditor/gv/zsocketlayout.h"
#include "nodeeditor/gv/zlayoutbackground.h"


class ZenoGraphsEditor;
class ZenoSubGraphScene;
class GroupNode;
class ParamsModel;
class StatusGroup;
class StatusButton;

class ZenoNodeBase : public ZLayoutBackground
{
    Q_OBJECT
    typedef ZLayoutBackground _base;

public:
    ZenoNodeBase(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    virtual ~ZenoNodeBase();
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    void initUI(const QModelIndex& index);

    enum { Type = ZTYPE_NODE };
    int type() const override;
    QString nodeId() const;
    QString nodeClass() const;
    QPointF nodePos() const;
    QPersistentModelIndex index() const { return m_index; }
    void markNodeStatus(zeno::NodeRunStatus status);
    void setMoving(bool isMoving);
    void setGroupNode(GroupNode* pNode);
    GroupNode* getGroupNode();

    virtual QModelIndex getSocketIndex(QGraphicsItem* uiitem, bool bSocketText) const;
    virtual QPointF getSocketPos(const QModelIndex& sockIdx, const QString keyName = "");
    virtual ZenoSocketItem* getNearestSocket(const QPointF& pos, bool bInput);
    virtual ZenoSocketItem* getSocketItem(const QModelIndex& sockIdx, const QString keyName);

    virtual void onZoomed();
    virtual void onCollaspeUpdated(bool);
    virtual void onCollaspeBtnClicked();
    virtual void onRunStateChanged() {};
    virtual void onSocketLinkChanged(const QModelIndex& paramIdx, bool bInput, bool bAdded, const QString keyName) {};
    virtual void onOptionsUpdated(int options) {};
    virtual void onViewUpdated(bool bView) {};
    virtual void setSelected(bool);

signals:
    void nodePosChangedSignal();
    void inSocketPosChanged();
    void outSocketPosChanged();
    void socketClicked(ZenoSocketItem*);

protected:
    virtual void initLayout() = 0;

    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void updateWhole();
    ZenoGraphsEditor* getEditorViewByViewport(QWidget* pWidget);

private:
    void _drawBorderWangStyle(QPainter* painter);

protected:
    NodeUtilParam m_renderParams;
    QPersistentModelIndex m_index;
    //when initui, zenonode should not emit any signals.
    bool m_bUIInited;
    bool m_bVisible;

private:
    GroupNode *m_groupNode;
    bool m_bMoving;     //pos change flag.
    QPointF m_lastMoving;    //last moving pos.
};

#endif