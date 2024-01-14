#include "zenosocketitem.h"
#include "zgraphicstextitem.h"
#include "zgraphicsnetlabel.h"
#include "style/zenostyle.h"
#include "uicommon.h"
#include "zassert.h"


ZenoSocketItem::ZenoSocketItem(
        const QPersistentModelIndex& viewSockIdx,
        const QSizeF& sz,
        QGraphicsItem* parent
)
    : _base(parent)
    , m_paramIdx(viewSockIdx)
    , m_status(STATUS_UNKNOWN)
    , m_size(sz)
    , m_bHovered(false)
    , m_netLabelItem(nullptr)
{
    m_bInput = m_paramIdx.data(ROLE_ISINPUT).toBool();
    m_innerSockMargin = ZenoStyle::dpiScaled(15);
    m_socketXOffset = ZenoStyle::dpiScaled(24);
    setData(GVKEY_SIZEHINT, m_size);
    setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    setSockStatus(STATUS_NOCONN);
    setAcceptHoverEvents(true);
}

int ZenoSocketItem::type() const
{
    return Type;
}

QPointF ZenoSocketItem::center() const
{
    //(0, 0) is the position of socket.
    QPointF center = mapToScene(QPointF(m_size.width() / 2., m_size.height() / 2.));
    return center;
}

QModelIndex ZenoSocketItem::paramIndex() const
{
    return m_paramIdx;
}

QRectF ZenoSocketItem::boundingRect() const
{
    QSizeF wholeSize = QSizeF(m_size.width() + m_socketXOffset, m_size.height());
    if (m_bInput)
        return QRectF(QPointF(-m_socketXOffset, 0), wholeSize);
    else
        return QRectF(QPointF(0, 0), wholeSize);
}

bool ZenoSocketItem::isInputSocket() const
{
    return m_bInput;
}

QString ZenoSocketItem::nodeIdent() const
{
    return m_paramIdx.isValid() ? m_paramIdx.data(ROLE_OBJID).toString() : "";
}

void ZenoSocketItem::setHovered(bool bHovered)
{
    m_bHovered = bHovered;
    update();
}

void ZenoSocketItem::setSockStatus(SOCK_STATUS status)
{
    if (m_status == status)
        return;

    m_status = status;
    update();
}

ZenoSocketItem::SOCK_STATUS ZenoSocketItem::sockStatus() const
{
    return m_status;
}

void ZenoSocketItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverEnterEvent(event);
    setHovered(true);
}

void ZenoSocketItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverMoveEvent(event);
}

void ZenoSocketItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverLeaveEvent(event);
    setHovered(false);
}

void ZenoSocketItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mousePressEvent(event);
    event->setAccepted(true);
}

void ZenoSocketItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
    update();
    if (this->isEnabled())
        emit clicked(m_bInput, event->button());
}

QString ZenoSocketItem::netLabel() const
{
    return m_netLabelItem ? m_netLabelItem->toPlainText() : "";
}

void ZenoSocketItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
#if 1
    if (editor_factor < 0.2)
        return;
#endif
    painter->setRenderHint(QPainter::Antialiasing, true);

    QColor bgClr;
    if (!isEnabled()) {
        bgClr = QColor(83, 83, 85);
    }
    else if (m_bHovered)
    {
        bgClr = QColor("#5FD2FF");
    }
    else
    {
        bgClr = QColor("#4B9EF4");
    }

    QPen pen(bgClr, ZenoStyle::dpiScaled(4));
    pen.setJoinStyle(Qt::RoundJoin);

    QColor innerBgclr(bgClr.red(), bgClr.green(), bgClr.blue(), 120);

    static const int startAngle = 0;
    static const int spanAngle = 360;
    painter->setPen(pen);
    bool bDrawBg = (m_status == STATUS_TRY_CONN || m_status == STATUS_CONNECTED || !netLabel().isEmpty());

    if (bDrawBg)
    {
        painter->setBrush(bgClr);
    }
    else
    {
        painter->setBrush(innerBgclr);
    }

    {
        QPainterPath path;
        qreal halfpw = pen.widthF() / 2;
        qreal xleft, xright, ytop, ybottom;

        xleft = halfpw;
        xright = m_size.width() - halfpw;
        ytop = halfpw;
        ybottom = m_size.height() - halfpw;

        if (m_bInput)
        {
            path.moveTo(QPointF(xleft, ybottom));
            path.lineTo(QPointF((xleft + xright) / 2., ybottom));
            //bottom right arc.
            QRectF rcBr(QPointF(xleft, (ytop + ybottom) / 2.), QPointF(xright, ybottom));
            path.arcTo(rcBr, 270, 90);

            QRectF rcTopRight(QPointF(xleft, ytop), QPointF(xright, (ybottom + ytop) / 2));
            path.lineTo(QPointF(xright, rcTopRight.center().y()));
            path.arcTo(rcTopRight, 0, 90);
            path.lineTo(QPointF(xleft, ytop));

            painter->setBrush(Qt::NoBrush);
            painter->drawPath(path);
            QRectF rc(QPointF(0, halfpw), QPointF(halfpw * 3.5, m_size.height() * 0.9));
            painter->fillRect(rc, bDrawBg ? bgClr : innerBgclr);
        }
        else
        {
            path.moveTo(QPointF(xright, ytop));
            path.lineTo(QPointF((xleft + xright) / 2., ytop));

            QRectF rcTopLeft(QPointF(xleft, ytop), QPointF(xright, (ytop + ybottom)/2.));
            path.arcTo(rcTopLeft, 90, 90);

            QRectF rcBottomLeft(QPointF(xleft, (ytop + ybottom) / 2.), QPointF(xright, ybottom));
            path.lineTo(QPointF(xleft, rcBottomLeft.center().y()));

            path.arcTo(rcBottomLeft, 180, 90);
            path.lineTo(QPointF(xright, ybottom));

            painter->setBrush(Qt::NoBrush);
            painter->drawPath(path);
            QRectF rc(QPointF(halfpw, halfpw), QPointF(m_size.width(), m_size.height() * 0.9));
            painter->fillRect(rc, bDrawBg ? bgClr : innerBgclr);
        }
    }
}