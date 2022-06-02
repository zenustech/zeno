#ifndef __ZENO_NODE_H__
#define __ZENO_NODE_H__

#include <QtWidgets>
#include <zenoui/render/renderparam.h>
#include <zenoui/nodesys/zenosvgitem.h>
#include "zenobackgrounditem.h"
#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/comctrl/gv/zenosocketitem.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenoui/model/modeldata.h>

class SubGraphModel;
class ZenoGraphsEditor;

class ZenoNode : public QGraphicsWidget
{
    Q_OBJECT
    typedef QGraphicsWidget _base;
    struct _socket_ctrl
    {
        ZenoSocketItem* socket;
        ZenoTextLayoutItem* socket_text;
        ZenoParamWidget* socket_control;

        _socket_ctrl() : socket(nullptr), socket_text(nullptr), socket_control(nullptr) {}
    };

public:
    ZenoNode(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    virtual ~ZenoNode();
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QRectF boundingRect() const override;

    enum { Type = ZTYPE_NODE };
    int type() const override;

    void initUI(const QModelIndex& subGIdx, const QModelIndex& index);
    void initLegacy(const QModelIndex& subGIdx, const QModelIndex& index);
    void initWangStyle(const QModelIndex& subGIdx, const QModelIndex& index);

    QPersistentModelIndex index() { return m_index; }
    QPointF getPortPos(bool bInput, const QString& portName);
    void toggleSocket(bool bInput, const QString& sockName, bool bSelected);
    void markError(bool isError);
    void getSocketInfoByItem(ZenoSocketItem* pSocketItem, QString& sockName, QPointF& scenePos, bool& bInput, QPersistentModelIndex& linkIdx);

    QString nodeId() const;
    QString nodeName() const;
    QPointF nodePos() const;
    INPUT_SOCKETS inputParams() const;
    OUTPUT_SOCKETS outputParams() const;

signals:
    void socketClicked(const QString& id, bool bInput, const QString& name);
    void doubleClicked(const QString &nodename);
    void paramChanged(const QString& nodeid, const QString& paramName, const QVariant& var);
    void socketPosInited(const QString& nodeid, const QString& sockName, bool bInput);
    void statusBtnHovered(STATUS_BTN);

public slots:
    void onCollaspeBtnClicked();
    void onCollaspeLegacyUpdated(bool);
    void onCollaspeUpdated(bool);
    void onOptionsBtnToggled(STATUS_BTN btn, bool toggled);
    void onOptionsUpdated(int options);
    void onParamUpdated(const QString &paramName, const QVariant &val);
    void onSocketUpdated(const SOCKET_UPDATE_INFO& info);
    void onSocketDeflUpdated(const PARAM_UPDATE_INFO& info);
    void onSocketLinkChanged(const QString& sockName, bool bInput, bool bAdded);
    void onSocketsUpdateOverall(bool bInput);
    void onInOutSocketChanged(bool bInput);
    void updateSocketDeflValue(const QString& nodeid, const QString& inSock, const INPUT_SOCKET& inSocket, const QVariant& textValue);
    void onNameUpdated(const QString& newName);

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
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF& constraint = QSizeF()) const override;
    //ZenoNode:
    virtual void onParamEditFinished(PARAM_CONTROL editCtrl, const QString& paramName, const QString& textValue);
    QPersistentModelIndex subGraphIndex() const;
    virtual QGraphicsLayout* initParams();
    virtual void initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout* pParamLayout, const QString& name, const PARAM_INFO& param);
    virtual QGraphicsLinearLayout* initCustomParamWidgets();

protected:
    NodeUtilParam m_renderParams;

private:
    ZenoBackgroundWidget* initBodyWidget(NODE_TYPE type);
    ZenoBackgroundWidget* initHeaderLegacy(NODE_TYPE type);
    ZenoBackgroundWidget* initHeaderWangStyle(NODE_TYPE type);
    ZenoBackgroundWidget* initCollaspedWidget();
    QGraphicsLayout* initSockets();
    void initIndependentWidgetsLegacy();
    void _initSocketItemPos();
    void _initStatusBtnPos();
    void _drawBorderWangStyle(QPainter* painter);
    ZenoGraphsEditor* getEditorViewByViewport(QWidget* pWidget);
    void updateWhole();

    QPersistentModelIndex m_index;
    QPersistentModelIndex m_subGpIndex;

    QMap<QString, _socket_ctrl> m_inSockets;
    QMap<QString, _socket_ctrl> m_outSockets;

    QMap<QString, ZenoParamWidget*> m_paramControls;

    QGraphicsTextItem* m_nameItem;
    ZenoTextLayoutItem* m_NameItem;
    ZenoImageItem *m_mute;
    ZenoImageItem *m_view;
    ZenoImageItem *m_once;
    ZenoImageItem *m_collaspe;

    ZenoBackgroundWidget* m_collaspedWidget;
    ZenoBackgroundWidget *m_bodyWidget;
    ZenoBackgroundWidget *m_headerWidget;
    ZenoMinStatusBtnWidget* m_pStatusWidgets;

    QGraphicsLinearLayout* m_pMainLayout;
    QGraphicsLinearLayout* m_pSocketsLayout;
    QGraphicsLinearLayout* m_pInSocketsLayout;
    QGraphicsLinearLayout* m_pOutSocketsLayout;
    QGraphicsRectItem* m_border;

    bool m_bInitSockets;
    bool m_bHeapMap;
    bool m_bError;
};

#endif