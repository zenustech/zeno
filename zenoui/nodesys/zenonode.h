#ifndef __ZENO_NODE_H__
#define __ZENO_NODE_H__

#include <QtWidgets>
#include "../render/renderparam.h"
#include "zenosvgitem.h"
#include "zenobackgrounditem.h"
#include "nodesys_common.h"
#include "zenosocketitem.h"
#include "zenoparamwidget.h"
#include "../model/modeldata.h"

class SubGraphModel;

class ZenoNode : public QGraphicsWidget
{
    Q_OBJECT
    typedef QGraphicsWidget _base;
public:
    ZenoNode(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    ~ZenoNode();
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QRectF boundingRect() const override;

    enum { Type = ZTYPE_NODE };
    int type() const override;

    void init(const QModelIndex& index, SubGraphModel* pModel);

    QPersistentModelIndex index() { return m_index; }
    QPointF getPortPos(bool bInput, const QString& portName);
    void toggleSocket(bool bInput, const QString& sockName, bool bSelected);

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

public slots:
    void onCollaspeBtnClicked();
    void onCollaspeUpdated(bool);
    void onOptionsUpdated(int options);
    void onParamUpdated(const QString &paramName, const QVariant &val);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
    bool sceneEvent(QEvent *event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void resizeEvent(QGraphicsSceneResizeEvent *event) override;

private:
    ZenoBackgroundWidget* initBodyWidget(NODE_TYPE type);
    ZenoBackgroundWidget* initHeaderBgWidget(NODE_TYPE type);
    ZenoBackgroundWidget* initCollaspedWidget();
    QGraphicsGridLayout* initParams();
    QGraphicsGridLayout* initSockets();
    void initIndependentWidgets();
    void _initSocketItemPos();

    QPersistentModelIndex m_index;
    NodeUtilParam m_renderParams;
    std::map<QString, ZenoSocketItem*> m_inSocks;
    std::map<QString, ZenoSocketItem*> m_outSocks;
    QMap<QString, ZenoTextLayoutItem*> m_inSockNames;
    QMap<QString, ZenoTextLayoutItem*> m_outSockNames;
    QMap<QString, ZenoParamWidget*> m_paramControls;

    QGraphicsTextItem* m_nameItem;
    ZenoImageItem *m_mute;
    ZenoImageItem *m_view;
    ZenoImageItem *m_prep;
    ZenoImageItem *m_collaspe;

    ZenoBackgroundWidget* m_collaspedWidget;
    ZenoBackgroundWidget *m_bodyWidget;
    ZenoBackgroundWidget *m_headerWidget;

    QGraphicsLinearLayout *m_pMainLayout;

    bool m_bInitSockets;
    bool m_bHeapMap;
};

#endif