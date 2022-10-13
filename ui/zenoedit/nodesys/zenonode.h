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

class ZenoNode : public ZLayoutBackground
{
    Q_OBJECT
    typedef ZLayoutBackground _base;
    struct _param_ctrl
    {
        ZSimpleTextItem* param_name;
        ZenoParamWidget* param_control;
        ZGraphicsLayout* ctrl_layout;
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
    QPointF getPortPos(bool bInput, const QString& portName);
    ZenoSocketItem* getNearestSocket(const QPointF& pos, bool bInput);
    ZenoSocketItem* getSocketItem(bool bInput, const QString& sockName);
    void setGeometry(const QRectF& rect) override;
    void toggleSocket(bool bInput, const QString& sockName, bool bSelected);
    void markError(bool isError);
    void getSocketInfoByItem(ZenoSocketItem* pSocketItem, QString& sockName, QPointF& scenePos, bool& bInput, QPersistentModelIndex& linkIdx);

    QString nodeId() const;
    QString nodeName() const;
    QPointF nodePos() const;
    INPUT_SOCKETS inputParams() const;
    OUTPUT_SOCKETS outputParams() const;
    virtual void onUpdateParamsNotDesc();

signals:
    void socketClicked(QString nodeid, bool bInput, QString sockName, QPointF scenePos, QPersistentModelIndex linkIndex);
    void doubleClicked(const QString &nodename);
    void paramChanged(const QString& nodeid, const QString& paramName, const QVariant& var);
    void socketPosInited(const QString& nodeid, const QString& sockName, bool bInput);
    void statusBtnHovered(STATUS_BTN);
    void inSocketPosChanged();
    void outSocketPosChanged();

public slots:
    void onCollaspeBtnClicked();
    void onCollaspeUpdated(bool);
    void onOptionsBtnToggled(STATUS_BTN btn, bool toggled);
    void onOptionsUpdated(int options);
    void onParamUpdated(ZenoSubGraphScene* pScene, const QString &paramName, const QVariant &val);
    void onSocketLinkChanged(const QString& sockName, bool bInput, bool bAdded);
    void onSocketsUpdate(ZenoSubGraphScene* pScene, bool bInput, bool bInit);
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
    virtual void onParamEditFinished(const QString& paramName, const QVariant& value);
    QPersistentModelIndex subGraphIndex() const;
    virtual ZLayoutBackground* initBodyWidget(ZenoSubGraphScene* pScene);
    virtual ZLayoutBackground* initHeaderWidget();
    virtual ZGraphicsLayout* initParam(PARAM_CONTROL ctrl, const QString& name, const PARAM_INFO& param, ZenoSubGraphScene* pScene);
    virtual ZGraphicsLayout* initCustomParamWidgets();
    virtual QValidator* validateForParams(PARAM_INFO info);
    virtual QValidator* validateForSockets(INPUT_SOCKET inSocket);

protected:
    NodeUtilParam m_renderParams;

    ZLayoutBackground* m_bodyWidget;
    ZLayoutBackground* m_headerWidget;

private:
    void _drawBorderWangStyle(QPainter* painter);
    ZenoGraphsEditor* getEditorViewByViewport(QWidget* pWidget);
    ZenoParamWidget* initSocketWidget(ZenoSubGraphScene* scene, const INPUT_SOCKET inSocket);
    ZenoParamWidget* initParamWidget(ZenoSubGraphScene* scene, const PARAM_INFO& param);
    bool renameDictKey(bool bInput, const INPUT_SOCKETS& inputs, const OUTPUT_SOCKETS& outputs);
    void updateSocketWidget(ZenoSubGraphScene* pScene, const INPUT_SOCKET inSocket);

    QPersistentModelIndex m_index;
    QPersistentModelIndex m_subGpIndex;

    FuckQMap<QString, ZSocketLayout*> m_inSockets;
    FuckQMap<QString, _param_ctrl> m_params;
    FuckQMap<QString, ZSocketLayout*> m_outSockets;

    ZSimpleTextItem* m_NameItem;
    ZenoMinStatusBtnWidget* m_pStatusWidgets;

    QGraphicsRectItem* m_border;
    ZGraphicsLayout* m_bodyLayout;

    //when initui, zenonode should not emit any signals.
    bool m_bUIInited;

    bool m_bError;
    bool m_bEnableSnap;

    // when zoom out the view, the view of node will be displayed as text with large size font.
    // it's convenient to view all nodes in big scale picture, but it also brings some problem.
    static const bool bEnableZoomPreview = false;
};

#endif