#include "zenolink.h"
#include "zenonode.h"
#include "zenosubgraphscene.h"
#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/render/common_id.h>
#include <zenoui/comctrl/gv/zenosocketitem.h>
#include <zenoui/style/zenostyle.h>
#include "../util/log.h"
#include "settings/zenosettingsmanager.h"


ZenoLink::ZenoLink(QGraphicsItem *parent)
    : _base(parent)
{
    connect(&ZenoSettingsManager::GetInstance(), &ZenoSettingsManager::valueChanged, this, [=](QString key) { 
        if (key == zsLinkLineShape) {
            update();
        }
    });
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
    bool bCurve = ZenoSettingsManager::GetInstance().getValue(zsLinkLineShape).toBool();
    if (bCurve)
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
            path.cubicTo(src.x() + dist, src.y(), dst.x() - dist, dst.y(), dst.x(), dst.y());
        }

        hasLastPath = true;
        lastSrcPos = src;
        lastDstPos = dst;
        lastPath = path;
        return path;
    } 
    else 
    {
        QPainterPath path;
        path.moveTo(getSrcPos());
        path.lineTo(getDstPos());
        return path;
    }
}

int ZenoLink::type() const
{
    return Type;
}

void ZenoLink::paint(QPainter* painter, QStyleOptionGraphicsItem const* styleOptions, QWidget* widget)
{
    painter->save();
    QPen pen;
    pen.setColor(isSelected() ? QColor(0xFA6400) : QColor("#4B9EF4"));
    pen.setWidthF(ZenoStyle::scaleWidth(WIDTH));
    painter->setRenderHint(QPainter::Antialiasing);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->drawPath(shape());
    painter->restore();
}


ZenoTempLink::ZenoTempLink(ZenoSocketItem* socketItem, QString nodeId, QPointF fixedPos, bool fixInput, QModelIndexList selNodes)
    : ZenoLink(nullptr)
    , m_fixedSocket(socketItem)
    , m_fixedPos(fixedPos)
    , m_floatingPos(fixedPos)
    , m_bfixInput(fixInput)
    , m_nodeId(nodeId)
    , m_adsortedSocket(nullptr)
    , m_selNodes(selNodes)
{
}

ZenoTempLink::~ZenoTempLink()
{
}

QPointF ZenoTempLink::getSrcPos() const
{
    return m_bfixInput ? m_floatingPos : m_fixedPos;
}

QPointF ZenoTempLink::getDstPos() const
{
    return m_bfixInput ? m_fixedPos : m_floatingPos;
}

void ZenoTempLink::setOldLink(const QPersistentModelIndex& link)
{
    m_oldLink = link;
}

QPersistentModelIndex ZenoTempLink::oldLink() const
{
    return m_oldLink;
}

QModelIndexList ZenoTempLink::selNodes() const
{
    return m_selNodes;
}

void ZenoTempLink::paint(QPainter* painter, QStyleOptionGraphicsItem const* styleOptions, QWidget* widget)
{
    painter->save();
    QPen pen;
    pen.setColor(QColor("#5FD2FF"));
    pen.setWidthF(ZenoStyle::scaleWidth(WIDTH));
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

void ZenoTempLink::getFixedInfo(QString& nodeId, QPointF& fixedPos, bool& bFixedInput)
{
    nodeId = m_nodeId;
    fixedPos = m_fixedPos;
    bFixedInput = m_bfixInput;
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
    {
        QModelIndex idx = m_adsortedSocket->paramIndex();
        PARAM_LINKS links = idx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
        if (links.isEmpty() || (links.size() == 1 && links[0] == m_oldLink))
            m_adsortedSocket->setSockStatus(ZenoSocketItem::STATUS_TRY_DISCONN);
    }
    m_adsortedSocket = pSocket;
    if (m_adsortedSocket)
    {
        m_adsortedSocket->setSockStatus(ZenoSocketItem::STATUS_TRY_CONN);
    }
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

    const QModelIndex& inSockIdx = m_index.data(ROLE_INSOCK_IDX).toModelIndex();
    const QModelIndex& outSockIdx = m_index.data(ROLE_OUTSOCK_IDX).toModelIndex();
    if (inSockIdx.data(ROLE_PARAM_CLASS) == PARAM_INNER_INPUT ||
        outSockIdx.data(ROLE_PARAM_CLASS) == PARAM_INNER_OUTPUT)
    {
        setZValue(ZVALUE_LINK_ABOVE);
    }
    else
    {
        setZValue(ZVALUE_LINK);
    }
    setFlag(QGraphicsItem::ItemIsSelectable);

    m_inNode = idx.data(ROLE_INNODE).toString();
    m_outNode = idx.data(ROLE_OUTNODE).toString();

    m_dstPos = inNode->getSocketPos(inSockIdx);
    m_srcPos = outNode->getSocketPos(outSockIdx);

    connect(inNode, SIGNAL(inSocketPosChanged()), this, SLOT(onInSocketPosChanged()));
    connect(outNode, SIGNAL(outSocketPosChanged()), this, SLOT(onOutSocketPosChanged()));
}

void ZenoFullLink::onInSocketPosChanged()
{
    if (!m_index.isValid())
        return;
    ZenoNode* pNode = qobject_cast<ZenoNode*>(sender());
    ZASSERT_EXIT(pNode);
    const QModelIndex& inSockIdx = m_index.data(ROLE_INSOCK_IDX).toModelIndex();
    m_dstPos = pNode->getSocketPos(inSockIdx);
}

void ZenoFullLink::onOutSocketPosChanged()
{
    if (!m_index.isValid())
        return;
    ZenoNode* pNode = qobject_cast<ZenoNode*>(sender());
    ZASSERT_EXIT(pNode);
    const QModelIndex& outSockIdx = m_index.data(ROLE_OUTSOCK_IDX).toModelIndex();
    m_srcPos = pNode->getSocketPos(outSockIdx);
}

QPersistentModelIndex ZenoFullLink::linkInfo() const
{
    return m_index;
}

QPainterPath ZenoFullLink::shape() const
{
    const QModelIndex& outSockIdx = m_index.data(ROLE_OUTSOCK_IDX).toModelIndex();
    if (outSockIdx.data(ROLE_VPARAM_REF).toBool())
    {
        QString text = outSockIdx.data(ROLE_OBJID).toString();
        QString socketName = outSockIdx.data(ROLE_PARAM_NAME).toString();
        text = text + ": " + socketName;
        QFontMetrics fontMetrics(QApplication::font());
        qreal width = fontMetrics.width(text);
        qreal dist = ZenoStyle::dpiScaled(8);
        QPainterPath path(m_srcPos + QPointF(0, dist));
        QPointF endPos(m_srcPos.x() + dist + width, m_srcPos.y() + dist);
        path.lineTo(endPos);
        path.moveTo(m_dstPos + QPointF(0, dist));
        endPos = QPointF(m_dstPos.x() - width - dist, m_dstPos.y() + dist);
        path.lineTo(endPos);
        return path;
    }
    return ZenoLink::shape();
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

void ZenoFullLink::paint(QPainter* painter, QStyleOptionGraphicsItem const* styleOptions, QWidget* widget)
{
    const QModelIndex& outSockIdx = m_index.data(ROLE_OUTSOCK_IDX).toModelIndex();
    if (outSockIdx.data(ROLE_VPARAM_REF).toBool())
    {
        QString text = outSockIdx.data(ROLE_OBJID).toString();
        QString socketName = outSockIdx.data(ROLE_PARAM_NAME).toString();
        text = text + ": " +socketName;
        QFontMetrics fontMetrics(QApplication::font());
        qreal width = fontMetrics.width(text);
        //draw line
        QPen pen(isSelected() ? QColor(0xFA6400) : QColor("#4B9EF4"), ZenoStyle::dpiScaled(3));
        painter->setPen(pen);
        painter->drawPath(shape());

        //draw outSocket text
        QPen textPen(QColor("#FFFFFF"), 1);
        painter->setPen(textPen);
        qreal x = m_srcPos.x() + ZenoStyle::dpiScaled(8);
        qreal y = m_srcPos.y() + ZenoStyle::dpiScaled(8) - fontMetrics.height() / 2;
        painter->drawText(x, y, text);

        //draw inSocket text
        x = m_dstPos.x() - width - ZenoStyle::dpiScaled(8);
        y = m_dstPos.y() + ZenoStyle::dpiScaled(8) - fontMetrics.height() / 2;
        painter->setPen(textPen);
        painter->drawText(x, y, text);

        return;
    }
    ZenoLink::paint(painter, styleOptions, widget);

}

int ZenoFullLink::type() const
{
    return Type;
}
