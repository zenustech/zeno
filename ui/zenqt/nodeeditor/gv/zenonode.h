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


class ZenoGraphsEditor;
class ZenoSubGraphScene;
class GroupNode;
class ParamsModel;

class ZenoNode : public ZLayoutBackground
{
    Q_OBJECT
    typedef ZLayoutBackground _base;

  public:
    struct _param_ctrl
    {
        ZSimpleTextItem* param_name;
        QGraphicsItem* param_control;
        ZGraphicsLayout* ctrl_layout;
        QPersistentModelIndex viewidx;
        _param_ctrl() : param_name(nullptr), param_control(nullptr), ctrl_layout(nullptr) {}
    };

public:
    ZenoNode(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    virtual ~ZenoNode();
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QRectF boundingRect() const override;

    enum { Type = ZTYPE_NODE };
    int type() const override;

    void initUI(ZenoSubGraphScene* pScene, const QModelIndex& subGIdx, const QModelIndex& index);

    QPersistentModelIndex index() { return m_index; }
    QPersistentModelIndex subgIndex() { return m_subGpIndex; }
    QModelIndex getSocketIndex(QGraphicsItem* uiitem, bool bSocketText) const;
    QPointF getSocketPos(const QModelIndex& sockIdx, const QString keyName = "");
    ZenoSocketItem* getNearestSocket(const QPointF& pos, bool bInput);
    ZenoSocketItem* getSocketItem(const QModelIndex& sockIdx, const QString keyName);
    void markError(bool isError);

    QString nodeId() const;
    QString nodeName() const;
    QPointF nodePos() const;
    void updateNodePos(const QPointF &pos, bool enableTransaction = true);
    virtual void onUpdateParamsNotDesc();
    void onMarkDataChanged(bool bDirty);

    void setMoving(bool isMoving);
    bool isMoving();

    virtual void onZoomed();
    void setGroupNode(GroupNode *pNode);
    GroupNode *getGroupNode();
    void addParam(const _param_ctrl &param);

    virtual void setSelected(bool selected);

signals:
    void socketClicked(ZenoSocketItem*, zeno::LinkFunction);
    void doubleClicked(const QString &nodename);
    void paramChanged(const QString& nodeid, const QString& paramName, const QVariant& var);
    void socketPosInited(const QString& nodeid, const QString& sockName, bool bInput);
    void statusBtnHovered(STATUS_BTN);
    void inSocketPosChanged();
    void outSocketPosChanged();
    void nodePosChangedSignal();

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
    QPersistentModelIndex subGraphIndex() const;
    virtual ZLayoutBackground* initBodyWidget(ZenoSubGraphScene* pScene);
    virtual ZLayoutBackground* initHeaderWidget();
    virtual ZGraphicsLayout* initSockets(ParamsModel* pModel, const bool bInput, ZenoSubGraphScene* pScene);
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
    ZGraphicsLayout* initVerticalSockets(bool bInput);
    void _drawBorderWangStyle(QPainter* painter);
    ZSocketLayout* getSocketLayout(bool bInput, const QString& sockName);
    ZSocketLayout* getSocketLayout(bool bInput, int idx);
    bool removeSocketLayout(bool bInput, const QString& sockName);
    void focusOnNode(const QModelIndex& nodeIdx);

    ZenoGraphsEditor* getEditorViewByViewport(QWidget* pWidget);
    QGraphicsItem* initSocketWidget(ZenoSubGraphScene* scene, const QModelIndex& paramIdx);
    void updateWhole();
    SocketBackgroud* addSocket(const QModelIndex& idx, bool bInput, ZenoSubGraphScene* pScene);
    void onUpdateFrame(QGraphicsItem* pContrl, int nFrame, QVariant val);
    void onPasteSocketRefSlot(QModelIndex toIndex);

    QVector<ZSocketLayout*> getSocketLayouts(bool bInput) const;

    QPersistentModelIndex m_index;
    QPersistentModelIndex m_subGpIndex;

    //QVector<ZSocketLayout*> m_inSockets;
    //QVector<ZSocketLayout*> m_outSockets;

    QKeyList<QString, _param_ctrl> m_params;

    ZGraphicsTextItem* m_NameItem;
    ZSimpleTextItem* m_pCategoryItem;
    ZSimpleTextItem *m_NameItemTip;
    ZenoMinStatusItem* m_pStatusWidgets;
    ZenoMinStatusItem* m_pStatusWidgets2;   //on collasped case.
    QGraphicsPolygonItem* m_statusMarker;

    QGraphicsRectItem* m_border;
    ZGraphicsLayout* m_bodyLayout;
    ZGraphicsLayout* m_inputsLayout;
    ZGraphicsLayout* m_outputsLayout;

    //when initui, zenonode should not emit any signals.
    bool m_bUIInited;

    bool m_bError;
    bool m_bEnableSnap;
    bool m_bMoving;     //pos change flag.
    QPointF m_lastMoving;    //last moving pos.

    // when zoom out the view, the view of node will be displayed as text with large size font.
    // it's convenient to view all nodes in big scale picture, but it also brings some problem.
    static const bool bEnableZoomPreview = false;
    static const bool bShowDataChanged = true;
    GroupNode *m_groupNode;
    bool m_bVisible;
};

#endif