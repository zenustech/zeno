#ifndef __ZENO_NODE_NEW_H__
#define __ZENO_NODE_NEW_H__

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
#include "nodeeditor/gv/zenonodebase.h"


class ZenoGraphsEditor;
class ZenoSubGraphScene;
class GroupNode;
class ParamsModel;
class StatusGroup;
class StatusButton;

class ZenoNodeNew : public ZenoNodeBase
{
    Q_OBJECT
    typedef ZenoNodeBase _base;

public:
    ZenoNodeNew(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    virtual ~ZenoNodeNew();
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

    void initLayout() override;

    QModelIndex getSocketIndex(QGraphicsItem* uiitem, bool bSocketText) const override;
    QPointF getSocketPos(const QModelIndex& sockIdx, const QString keyName = "") override;
    ZenoSocketItem* getNearestSocket(const QPointF& pos, bool bInput) override;
    ZenoSocketItem* getSocketItem(const QModelIndex& sockIdx, const QString keyName) override;
    ZenoSocketItem* getObjSocketItem(const QModelIndex& sockIdx, bool bInput);
    virtual void onZoomed() override;

public slots:
    void onCollaspeBtnClicked();
    void onCollaspeUpdated(bool);
    void onRunStateChanged();
    void onOptionsBtnToggled(STATUS_BTN btn, bool toggled);
    void onOptionsUpdated(int options);
    void onViewUpdated(bool bView);
    void onSocketLinkChanged(const QModelIndex& paramIdx, bool bInput, bool bAdded, const QString keyName);
    void onNameUpdated(const QString& newName);
    void onParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
    void onParamInserted(const QModelIndex& parent, int first, int last);
    void onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last);
    void onViewParamsMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row);
    void onLayoutChanged();

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;
    //ZenoNodeNew:
    virtual ZGraphicsLayout* initCustomParamWidgets();

private slots:
    void onCustomNameChanged();

private:
    ZLayoutBackground* initBodyWidget();
    ZLayoutBackground* initHeaderWidget();
    ZGraphicsLayout* initSockets(ParamsModel* pModel, const bool bInput);
    ZGraphicsLayout* initVerticalSockets(bool bInput);
    void _drawShadow(QPainter* painter);

    bool removeSocketLayout(bool bInput, const QString& sockName);
    void addOnlySocketToLayout(ZGraphicsLayout* pSocketLayout, const QModelIndex& paramIdx);
    SocketBackgroud* addSocket(const QModelIndex& idx, bool bInput);
    void markNodeStatus(zeno::NodeRunStatus status);

    QVector<ZSocketLayout*> getSocketLayouts(bool bInput) const;
    QVector<ZenoSocketItem*> getSocketItems(bool bInput) const;
    QVector<ZenoSocketItem*> getObjSocketItems(bool bInput) const;
    ZSocketLayout* getSocketLayout(bool bInput, const QString& sockName) const;
    ZSocketLayout* getSocketLayout(bool bInput, int idx) const;

private:
    ZGraphicsTextItem* m_NameItem;
    ZSimpleTextItem* m_pCategoryItem;
    ZSimpleTextItem *m_NameItemTip;
    StatusGroup* m_pStatusWidgets;
    ZenoImageItem* m_errorTip;
    QGraphicsPolygonItem* m_statusMarker;
    ZLayoutBackground* m_bodyWidget;
    ZLayoutBackground* m_headerWidget;

    ZGraphicsLayout* m_bodyLayout;
    ZGraphicsLayout* m_inputObjSockets;
    ZGraphicsLayout* m_outputObjSockets;
    ZGraphicsLayout* m_inputsLayout;
    ZGraphicsLayout* m_outputsLayout;

    zeno::NodeRunStatus m_nodeStatus;
};

#endif