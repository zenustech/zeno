#ifndef __ZENO_LINK_H__
#define __ZENO_LINK_H__

#include <QtWidgets>
//#include "uicommon.h"
#include "nodeeditor/gv/nodesys_common.h"
#include <zeno/core/common.h>

class ZenoSubGraphScene;
class ZenoSocketItem;
class ZenoNodeBase;

//#define BASE_ON_CURVE

class ZenoLink : public QGraphicsObject
{
    Q_OBJECT
    typedef QGraphicsObject _base;

public:
    ZenoLink(QGraphicsItem* parent = nullptr);
    virtual ~ZenoLink();

    virtual QRectF boundingRect() const override;
    virtual QPainterPath shape() const override;
    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;

    enum { Type = ZTYPE_LINK };
    int type() const override;

    virtual QPointF getSrcPos() const = 0;
    virtual QPointF getDstPos() const = 0;

protected:
    bool m_bothCollaspedNode = false;

private:
    static constexpr float BEZIER = 0.5f, WIDTH = 1;

    mutable QPointF lastSrcPos, lastDstPos;
    mutable bool hasLastPath{false};
    mutable QPainterPath lastPath;
};

class ZenoTempLink : public ZenoLink
{
    Q_OBJECT
public:
    ZenoTempLink(ZenoSocketItem* socketItem,
        QString nodeId,
        QPointF fixedPos,
        bool fixInput,
        QModelIndexList selNodes);
    ~ZenoTempLink();
    virtual QPointF getSrcPos() const override;
    virtual QPointF getDstPos() const override;
    void setFloatingPos(QPointF pos);
    void getFixedInfo(QString& nodeId, QPointF& fixedPos, bool& bFixedInput, bool lnkObj);
    ZenoSocketItem* getFixedSocket() const;
    ZenoSocketItem* getAdsorbedSocket() const;
    void setAdsortedSocket(ZenoSocketItem* pSocket);
    void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;
    void setOldLink(const QPersistentModelIndex& link);
    QPersistentModelIndex oldLink() const;
    QModelIndexList selNodes() const;

protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;

    enum { Type = ZTYPE_TEMPLINK };
    int type() const override;

private:
    static constexpr float BEZIER = 0.5f, WIDTH = 3;

    QString m_nodeId;
    QPointF m_floatingPos;
    QPointF m_fixedPos;
    ZenoSocketItem* m_adsortedSocket;
    ZenoSocketItem* m_fixedSocket;
    bool m_bObjLink;
    QPersistentModelIndex m_oldLink;    //the link which belongs to
    QModelIndexList m_selNodes;
    bool m_bfixInput;
};

class ZenoFullLink : public ZenoLink
{
    Q_OBJECT

public:
    ZenoFullLink(const QPersistentModelIndex& idx, ZenoNodeBase* outNode, ZenoNodeBase* inNode);
    ~ZenoFullLink();

    virtual QPointF getSrcPos() const override;
    virtual QPointF getDstPos() const override;
    QPersistentModelIndex linkInfo() const;
    QPainterPath shape() const override;

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;

    enum { Type = ZTYPE_FULLLINK };
    int type() const override;

    void paint(QPainter* painter, QStyleOptionGraphicsItem const* styleOptions, QWidget* widget) override;
    bool isLegacyLink() const;

protected:
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;

private slots:
    void onInSocketPosChanged();
    void onOutSocketPosChanged();

private:
    void focusOnNode(const QModelIndex &nodeIdx);
    bool isPrimLink();
    void getConnectedState(zeno::SocketType& inSockProp, bool& inNodeCollasped);

    QPersistentModelIndex m_index;
    QPointF m_srcPos, m_dstPos;
    QString m_inNode;
    QString m_outNode;
    bool m_bHover;
    bool m_bLegacyLink;
    bool m_bObjLink;
};

#endif
