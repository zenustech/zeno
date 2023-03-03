#ifndef __ZENO_NODE_H__
#define __ZENO_NODE_H__

#include <QtWidgets>
#include <zenoui/render/renderparam.h>
#include <zenoui/nodesys/zenosvgitem.h>
#include "zenobackgrounditem.h"
#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/comctrl/gv/zenosocketitem.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenomodel/include/modeldata.h>
#include <zenoui/comctrl/gv/zgraphicslayout.h>
#include <zenoui/comctrl/gv/zgraphicslayoutitem.h>
#include <zenoui/comctrl/gv/zsocketlayout.h>
#include <zenoui/comctrl/gv/zlayoutbackground.h>


class ZenoGraphsEditor;
class ZenoSubGraphScene;
class GroupNode;

class ZenoNode : public ZLayoutBackground
{
    Q_OBJECT
    typedef ZLayoutBackground _base;
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
    QPointF getSocketPos(const QModelIndex& sockIdx);
    ZenoSocketItem* getNearestSocket(const QPointF& pos, bool bInput);
    ZenoSocketItem* getSocketItem(const QModelIndex& sockIdx);
    void markError(bool isError);

    QString nodeId() const;
    QString nodeName() const;
    QPointF nodePos() const;
    void updateNodePos(const QPointF &pos, bool enableTransaction = true);
    virtual void onUpdateParamsNotDesc();

    void setMoving(bool isMoving);
    bool isMoving();

    virtual void onZoomed();
    void setGroupNode(GroupNode *pNode);
    GroupNode *getGroupNode();

signals:
    void socketClicked(ZenoSocketItem*);
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
    void onOptionsBtnToggled(STATUS_BTN btn, bool toggled);
    void onOptionsUpdated(int options);
    void onSocketLinkChanged(const QModelIndex& paramIdx, bool bInput, bool bAdded);
    void onNameUpdated(const QString& newName);
    void onViewParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
    void onViewParamInserted(const QModelIndex& parent, int first, int last);
    void onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last);

    void onViewParamAboutToBeMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row);
    void onViewParamsMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
    bool sceneEvent(QEvent *event) override;
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void resizeEvent(QGraphicsSceneResizeEvent *event) override;
	void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
	void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
	void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;
    void focusOutEvent(QFocusEvent *event) override;
    //ZenoNode:
    QPersistentModelIndex subGraphIndex() const;
    virtual ZLayoutBackground* initBodyWidget(ZenoSubGraphScene* pScene);
    virtual ZLayoutBackground* initHeaderWidget(IGraphsModel* pGraphsModel);
    virtual ZGraphicsLayout* initSockets(QStandardItem* socketItems, const bool bInput, ZenoSubGraphScene* pScene);
    virtual ZGraphicsLayout* initParams(QStandardItem* paramItems, ZenoSubGraphScene* pScene);
    virtual ZGraphicsLayout* initCustomParamWidgets();

protected:
    NodeUtilParam m_renderParams;

    ZLayoutBackground* m_bodyWidget;
    ZLayoutBackground* m_headerWidget;

private:
    void _drawBorderWangStyle(QPainter* painter);
    ZSocketLayout* getSocketLayout(bool bInput, const QString& sockName);
    bool removeSocketLayout(bool bInput, const QString& sockName);

    ZenoGraphsEditor* getEditorViewByViewport(QWidget* pWidget);
    QGraphicsItem* initSocketWidget(ZenoSubGraphScene* scene, const QModelIndex& paramIdx);
    QGraphicsItem* initParamWidget(ZenoSubGraphScene* scene, const QModelIndex& paramIdx);
    void updateWhole();
    ZSocketLayout* addSocket(const QModelIndex& idx, bool bInput, ZenoSubGraphScene* pScene);
    ZGraphicsLayout* addParam(const QModelIndex& idx, ZenoSubGraphScene* pScene);

    QPersistentModelIndex m_index;
    QPersistentModelIndex m_subGpIndex;

#if 0
    FuckQMap<QString, ZSocketLayout*> m_inSockets;
#endif
    QVector<ZSocketLayout*> m_inSockets;

    FuckQMap<QString, _param_ctrl> m_params;
    QVector<ZSocketLayout*> m_outSockets;

    ZSimpleTextItem* m_NameItem;
    ZenoMinStatusBtnItem* m_pStatusWidgets;

    QGraphicsRectItem* m_border;
    ZGraphicsLayout* m_bodyLayout;
    ZGraphicsLayout* m_inputsLayout;
    ZGraphicsLayout* m_paramsLayout;
    ZGraphicsLayout* m_outputsLayout;

    //when initui, zenonode should not emit any signals.
    bool m_bUIInited;

    bool m_bError;
    bool m_bEnableSnap;
    bool m_bMoving;     //pos change flag.
    QPointF m_lastMovig;    //last moving pos.

    // when zoom out the view, the view of node will be displayed as text with large size font.
    // it's convenient to view all nodes in big scale picture, but it also brings some problem.
    static const bool bEnableZoomPreview = false;
    GroupNode *m_groupNode;
};

#endif