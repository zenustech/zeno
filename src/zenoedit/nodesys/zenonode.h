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

class ZenoNode : public QGraphicsWidget
{
    Q_OBJECT
    typedef QGraphicsWidget _base;
    struct _socket_ctrl
    {
        ZenoSocketItem* socket;
        ZenoTextLayoutItem* socket_text;
    };

public:
    ZenoNode(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    ~ZenoNode();
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
    void getSocketInfoByItem(ZenoSocketItem* pSocketItem, QString& sockName, QPointF& scenePos, bool& bInput);

    QString nodeId() const;
    QString nodeName() const;
    QPointF nodePos() const;
    INPUT_SOCKETS inputParams() const;
    OUTPUT_SOCKETS outputParams() const;

signals:
    void nodePositionChange(const QString&);
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
    void onNameUpdated(const QString& newName);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
    bool sceneEvent(QEvent *event) override;
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

private:
    ZenoBackgroundWidget* initBodyWidget(NODE_TYPE type);
    ZenoBackgroundWidget* initHeaderLegacy(NODE_TYPE type);
    ZenoBackgroundWidget* initHeaderWangStyle(NODE_TYPE type);
    ZenoBackgroundWidget* initCollaspedWidget();
    QGraphicsLayout* initParams();
    QGraphicsGridLayout* initSockets();
    void initIndependentWidgetsLegacy();
    void _initSocketItemPos();
    void _initStatusBtnPos();
    void _drawBorderWangStyle(QPainter* painter);

    QPersistentModelIndex m_index;
    QPersistentModelIndex m_subGpIndex;
    NodeUtilParam m_renderParams;
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

    QGraphicsLinearLayout *m_pMainLayout;
    QGraphicsRectItem* m_border;

    bool m_bInitSockets;
    bool m_bHeapMap;
};

#endif