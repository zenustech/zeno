#include "zenolink.h"
#include "zenonode.h"
#include "zenosubgraphscene.h"
#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/render/common_id.h>
#include <zenoui/comctrl/gv/zenosocketitem.h>
#include "../util/log.h"


ZenoLink::ZenoLink(QGraphicsItem *parent)
    : _base(parent)
{
}

ZenoLink::~ZenoLink()
{
}

QRectF ZenoLink::boundingRect() const
{
    return shape().boundingRect();
}

QPainterPath ZenoLink::shape() const
{
    auto src = getSrcPos();
    auto dst = getDstPos();
    if (hasLastPath && src == lastSrcPos && dst == lastSrcPos)
        return lastPath;

    QPainterPath path(src);
    if (BEZIER == 0) {
        path.lineTo(dst);
    } else {
        float dist = dst.x() - src.x();
        dist = std::clamp(std::abs(dist), 40.f, 700.f) * BEZIER;
        path.cubicTo(src.x() + dist, src.y(),
                     dst.x() - dist, dst.y(),
                     dst.x(), dst.y());
    }

    hasLastPath = true;
    lastSrcPos = src;
    lastDstPos = dst;
    lastPath = path;
    return path;
}

int ZenoLink::type() const
{
    return Type;
}

void ZenoLink::paint(QPainter* painter, QStyleOptionGraphicsItem const* styleOptions, QWidget* widget)
{
    painter->save();
    QPen pen;
    pen.setColor(QColor(isSelected() ? 0xFA6400 : 0x808080));
    pen.setWidthF(WIDTH);
    painter->setRenderHint(QPainter::Antialiasing);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->drawPath(shape());
    painter->restore();
}


ZenoTempLink::ZenoTempLink(ZenoSocketItem* socketItem, QString nodeId, QString sockName, QPointF fixedPos, bool fixInput)
    : ZenoLink(nullptr)
    , m_fixedSocket(socketItem)
    , m_fixedPos(fixedPos)
    , m_floatingPos(fixedPos)
    , m_bfixInput(fixInput)
    , m_nodeId(nodeId)
    , m_sockName(sockName)
    , m_adsortedSocket(nullptr)
{
    m_fixedSocket->setSockStatus(ZenoSocketItem::STATUS_TRY_CONN);
}

ZenoTempLink::~ZenoTempLink()
{
    m_fixedSocket->setSockStatus(ZenoSocketItem::STATUS_TRY_DISCONN);
}

QPointF ZenoTempLink::getSrcPos() const
{
    return m_bfixInput ? m_floatingPos : m_fixedPos;
}

QPointF ZenoTempLink::getDstPos() const
{
    return m_bfixInput ? m_fixedPos : m_floatingPos;
}

void ZenoTempLink::paint(QPainter* painter, QStyleOptionGraphicsItem const* styleOptions, QWidget* widget)
{
    painter->save();
    QPen pen;
    pen.setColor(QColor(255,255,255));
    pen.setWidthF(WIDTH);
    painter->setRenderHint(QPainter::Antialiasing);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->drawPath(shape());
    painter->restore();
}

void ZenoTempLink::setFloatingPos(QPointF pos)
{
    m_floatingPos = pos;
    update();
}

void ZenoTempLink::getFixedInfo(QString& nodeId, QString& sockName, QPointF& fixedPos, bool& bFixedInput)
{
    nodeId = m_nodeId;
    fixedPos = m_fixedPos;
    bFixedInput = m_bfixInput;
    sockName = m_sockName;
}

ZenoSocketItem* ZenoTempLink::getAdsorbedSocket() const
{
    return m_adsortedSocket;
}

ZenoSocketItem* ZenoTempLink::getFixedSocket() const
{
    return m_fixedSocket;
}

void ZenoTempLink::setAdsortedSocket(ZenoSocketItem* pSocket)
{
    if (m_adsortedSocket)
        m_adsortedSocket->setSockStatus(ZenoSocketItem::STATUS_TRY_DISCONN);
    m_adsortedSocket = pSocket;
    if (m_adsortedSocket)
        m_adsortedSocket->setSockStatus(ZenoSocketItem::STATUS_TRY_CONN);
}

int ZenoTempLink::type() const
{
    return Type;
}

void ZenoTempLink::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ZenoLink::mouseMoveEvent(event);
    m_floatingPos = this->scenePos();
}


ZenoFullLink::ZenoFullLink(const QPersistentModelIndex& idx, ZenoNode* outNode, ZenoNode* inNode)
    : ZenoLink(nullptr)
    , m_index(idx)
{
    ZASSERT_EXIT(inNode && outNode && idx.isValid());

    setZValue(ZVALUE_LINK);
    setFlag(QGraphicsItem::ItemIsSelectable);

    m_inNode = idx.data(ROLE_INNODE).toString();
    m_outNode = idx.data(ROLE_OUTNODE).toString();
    m_inSock = idx.data(ROLE_INSOCK).toString();
    m_outSock = idx.data(ROLE_OUTSOCK).toString();

    m_srcPos = outNode->getPortPos(false, m_outSock);
    m_dstPos = inNode->getPortPos(true, m_inSock);

    connect(inNode, SIGNAL(inSocketPosChanged()), this, SLOT(onInSocketPosChanged()));
    connect(outNode, SIGNAL(outSocketPosChanged()), this, SLOT(onOutSocketPosChanged()));
}

void ZenoFullLink::onInSocketPosChanged()
{
    ZenoNode* pNode = qobject_cast<ZenoNode*>(sender());
    ZASSERT_EXIT(pNode);
    m_dstPos = pNode->getPortPos(true, m_inSock);
}

void ZenoFullLink::onOutSocketPosChanged()
{
    ZenoNode* pNode = qobject_cast<ZenoNode *>(sender());
    ZASSERT_EXIT(pNode);
    m_srcPos = pNode->getPortPos(false, m_outSock);
}

QPersistentModelIndex ZenoFullLink::linkInfo() const
{
    return m_index;
}

QPointF ZenoFullLink::getSrcPos() const
{
    return m_srcPos;
}

QPointF ZenoFullLink::getDstPos() const
{
    return m_dstPos;
}

void ZenoFullLink::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ZenoLink::mousePressEvent(event);
}

int ZenoFullLink::type() const
{
    return Type;
}
