#ifndef __ZENO_LINK_H__
#define __ZENO_LINK_H__

#include <QtWidgets>
#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/model/modeldata.h>

class ZenoSubGraphScene;
class ZenoSocketItem;
class ZenoNode;

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

private:
    static constexpr float BEZIER = 0.5f, WIDTH = 3;

    mutable QPointF lastSrcPos, lastDstPos;
    mutable bool hasLastPath{false};
    mutable QPainterPath lastPath;
};

class ZenoTempLink : public ZenoLink
{
    Q_OBJECT
public:
    ZenoTempLink(ZenoSocketItem* socketItem, QString nodeId, QString sockName, QPointF fixedPos, bool fixInput);
    ~ZenoTempLink();
    virtual QPointF getSrcPos() const override;
    virtual QPointF getDstPos() const override;
    void setFloatingPos(QPointF pos);
    void getFixedInfo(QString& nodeId, QString& sockName, QPointF& fixedPos, bool& bFixedInput);
    ZenoSocketItem* getFixedSocket() const;
    ZenoSocketItem* getAdsorbedSocket() const;
    void setAdsortedSocket(ZenoSocketItem* pSocket);
    void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget);

protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;

    enum { Type = ZTYPE_TEMPLINK };
    int type() const override;

private:
    static constexpr float BEZIER = 0.5f, WIDTH = 3;

    QString m_sockName;
    QString m_nodeId;
    QPointF m_floatingPos;
    SOCKET_INFO m_info;
    QPointF m_fixedPos;
    ZenoSocketItem* m_adsortedSocket;
    ZenoSocketItem* m_fixedSocket;
    bool m_bfixInput;
};

class ZenoFullLink : public ZenoLink
{
    Q_OBJECT
public:
    ZenoFullLink(const QPersistentModelIndex& idx, ZenoNode* outNode, ZenoNode* inNode);

    virtual QPointF getSrcPos() const override;
    virtual QPointF getDstPos() const override;
    QPersistentModelIndex linkInfo() const;

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

    enum { Type = ZTYPE_FULLLINK };
    int type() const override;

private slots:
    void onInSocketPosChanged();
    void onOutSocketPosChanged();

private:
    QPersistentModelIndex m_index;
    QPointF m_srcPos, m_dstPos;
    QString m_inNode;
    QString m_outNode;
};

#endif
