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

    void init(const QModelIndex& index);
    QGraphicsGridLayout* initParams();
    QGraphicsGridLayout* initSockets();

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

public slots:
    void onCollaspeBtnClicked();
    void collaspe(bool);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
    bool sceneEvent(QEvent* event) override;

private:
    ZenoBackgroundWidget* initBodyWidget();
    ZenoBackgroundWidget* initHeaderBgWidget();
    ZenoBackgroundWidget* initCollaspedWidget();
    void initIndependentWidgets();
    void _updateSocketItemPos();

    QPersistentModelIndex m_index;
    NodeUtilParam m_renderParams;
    std::map<QString, ZenoSocketItem*> m_inSocks;
    std::map<QString, ZenoSocketItem*> m_outSocks;
    QMap<QString, ZenoTextLayoutItem*> m_inSockNames;
    QMap<QString, ZenoTextLayoutItem*> m_outSockNames;

    QGraphicsTextItem* m_nameItem;
    ZenoImageItem *m_mute;
    ZenoImageItem *m_view;
    ZenoImageItem *m_prep;
    ZenoImageItem *m_collaspe;

    ZenoBackgroundWidget* m_collaspedWidget;
    ZenoBackgroundWidget *m_bodyWidget;
    ZenoBackgroundWidget *m_headerWidget;

    bool m_bInitSockets;
    bool m_bCollasped;
    bool m_bHeapMap;
};

#endif