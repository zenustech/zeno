#ifndef __ZENO_NODE_H__
#define __ZENO_NODE_H__

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
#include "zenonodebase.h"

class ZenoGraphsEditor;
class ZenoSubGraphScene;
class GroupNode;
class ParamsModel;
class StatusGroup;
class StatusButton;

class ZenoNode : public ZenoNodeBase
{
    Q_OBJECT
    typedef ZenoNodeBase _base;

public:
    ZenoNode(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    virtual ~ZenoNode();
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QRectF boundingRect() const override;

    void initLayout();

    QModelIndex getSocketIndex(QGraphicsItem* uiitem, bool bSocketText) const;
    QPointF getSocketPos(const QModelIndex& sockIdx, const QString keyName = "");
    ZenoSocketItem* getNearestSocket(const QPointF& pos, bool bInput);
    ZenoSocketItem* getSocketItem(const QModelIndex& sockIdx, const QString keyName);
    ZenoSocketItem* getTopBottomSocketItem(const QModelIndex& sockIdx, bool bInput);
    void markNodeStatus(zeno::NodeRunStatus status);

    void updateNodePos(const QPointF &pos, bool enableTransaction = true);
    virtual void onUpdateParamsNotDesc();
    void onMarkDataChanged(bool bDirty);

    virtual void onZoomed();
    //void addParam(const _param_ctrl &param);

    //virtual void setSelected(bool selected);

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
    void onViewParamAboutToBeMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row);
    void onViewParamsMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row);
    void onLayoutAboutToBeChanged();
    void onLayoutChanged();

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
    bool sceneEvent(QEvent *event) override;
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void resizeEvent(QGraphicsSceneResizeEvent *event) override;
	void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
	void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
	void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;
    void focusOutEvent(QFocusEvent *event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;
    //ZenoNode:
    virtual ZLayoutBackground* initBodyWidget();
    virtual ZLayoutBackground* initHeaderWidget();
    virtual ZGraphicsLayout* initSockets(ParamsModel* pModel, const bool bInput);
    virtual ZGraphicsLayout* initCustomParamWidgets();
    virtual Callback_OnButtonClicked registerButtonCallback(const QModelIndex& paramIdx);

protected:
    NodeUtilParam m_renderParams;

    ZLayoutBackground* m_bodyWidget;
    ZLayoutBackground* m_headerWidget;

    ZLayoutBackground* m_mainHeaderBg;

    ZGraphicsLayout* m_topInputSockets;
    ZGraphicsLayout* m_bottomOutputSockets;

private slots:
    void onCustomNameChanged();

private:
    ZLayoutBackground* initMainHeaderBg();
    ZGraphicsLayout* initNameLayout();
    ZGraphicsLayout* initVerticalSockets(bool bInput);
    ZSocketLayout* getSocketLayout(bool bInput, const QString& sockName);
    ZSocketLayout* getSocketLayout(bool bInput, int idx);
    bool removeSocketLayout(bool bInput, const QString& sockName);
    void focusOnNode(const QModelIndex& nodeIdx);
    void _drawShadow(QPainter* painter);
    void addOnlySocketToLayout(ZGraphicsLayout* pSocketLayout, const QModelIndex& paramIdx);

    ZenoGraphsEditor* getEditorViewByViewport(QWidget* pWidget);
    QGraphicsItem* initSocketWidget(const QModelIndex& paramIdx);
    void updateWhole();
    SocketBackgroud* addSocket(const QModelIndex& idx, bool bInput);
    void onUpdateFrame(QGraphicsItem* pContrl, int nFrame, QVariant val);
    void onPasteSocketRefSlot(QModelIndex toIndex);

    QVector<ZSocketLayout*> getSocketLayouts(bool bInput) const;
    QVector<ZenoSocketItem*> getSocketItems(bool bInput) const;

    //QVector<ZSocketLayout*> m_inSockets;
    //QVector<ZSocketLayout*> m_outSockets;

    ZGraphicsTextItem* m_NameItem;
    ZSimpleTextItem* m_pCategoryItem;
    ZSimpleTextItem *m_NameItemTip;
    StatusGroup* m_pStatusWidgets;
    StatusGroup* m_pMainStatusWidgets;
    ZenoImageItem* m_errorTip;
    //StatusGroup* m_pStatusWidgets2;
    QGraphicsPolygonItem* m_statusMarker;

    //QGraphicsRectItem* m_border;
    ZGraphicsLayout* m_expandNameLayout;
    ZGraphicsLayout* m_bodyLayout;
    ZGraphicsLayout* m_inputsLayout;
    ZGraphicsLayout* m_outputsLayout;

    zeno::NodeRunStatus m_nodeStatus;
    QString m_dbgName;      //only used to debug.
};

#endif