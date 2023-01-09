#include "zenosocketitem.h"
#include "zgraphicstextitem.h"
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/viewparammodel.h>


ZenoSocketItem::ZenoSocketItem(
        const QPersistentModelIndex& viewSockIdx,
        const QSizeF& sz,
        QGraphicsItem* parent
)
    : _base(parent)
    , m_viewSockIdx(viewSockIdx)
    , m_status(STATUS_UNKNOWN)
    , m_size(sz)
    , m_bgClr(QColor("#1992D7"))
{
    PARAM_CLASS cls = (PARAM_CLASS)viewSockIdx.data(ROLE_PARAM_CLASS).toInt();
    ZASSERT_EXIT(cls == PARAM_INNER_INPUT || cls == PARAM_INPUT ||
                 cls == PARAM_INNER_OUTPUT || cls == PARAM_OUTPUT);
    m_bInput = (cls == PARAM_INNER_INPUT || cls == PARAM_INPUT);
    m_bInnerSock = (cls == PARAM_INNER_INPUT || cls == PARAM_INNER_OUTPUT);
    m_margin = ZenoStyle::dpiScaled(15);
    setSockStatus(STATUS_NOCONN);
    setAcceptHoverEvents(true);
}

int ZenoSocketItem::type() const
{
    return Type;
}

QPointF ZenoSocketItem::center() const
{
    return this->sceneBoundingRect().center();
}

QModelIndex ZenoSocketItem::paramIndex() const
{
    return m_viewSockIdx;
}

QRectF ZenoSocketItem::boundingRect() const
{
    if (m_bInnerSock)
    {
        QRectF rc(QPointF(0, 0), m_size + QSize(2 * m_margin, 2 * m_margin));
        return rc;
    }
    else
    {
        return QRectF(QPointF(0, 0), m_size);
    }
}

bool ZenoSocketItem::isInputSocket() const
{
    return m_bInput;
}

QString ZenoSocketItem::nodeIdent() const
{
    return m_viewSockIdx.isValid() ? m_viewSockIdx.data(ROLE_OBJID).toString() : "";
}

void ZenoSocketItem::setSockStatus(SOCK_STATUS status)
{
    if (m_status == status)
        return;

    if (status == STATUS_NOCONN || status == STATUS_TRY_DISCONN)
    {
        if (m_viewSockIdx.isValid())
        {
            PARAM_LINKS links = m_viewSockIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
            if (!links.isEmpty())
                status = STATUS_CONNECTED;
        }
    }

    m_status = status;
    update();
}

void ZenoSocketItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    m_bgClr = QColor("#5FD2FF");
    _base::hoverEnterEvent(event);
}

void ZenoSocketItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverMoveEvent(event);
}

void ZenoSocketItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    m_bgClr = QColor("#1992D7");
    _base::hoverLeaveEvent(event);
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
    emit clicked(m_bInput);
}

void ZenoSocketItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    QPen pen(m_bgClr, 4);
    pen.setJoinStyle(Qt::RoundJoin);

    QColor innerBgclr(m_bgClr.red(), m_bgClr.green(), m_bgClr.blue(), 120);

    painter->setRenderHint(QPainter::Antialiasing, true);

    static const int startAngle = 0;
    static const int spanAngle = 360;
    painter->setPen(pen);
    bool bDrawBg = (m_status == STATUS_TRY_CONN || m_status == STATUS_CONNECTED);

    if (bDrawBg)
    {
        painter->setBrush(m_bgClr);
    }
    else
    {
        painter->setBrush(innerBgclr);
    }

    if (!m_bInnerSock)
    {
        QPainterPath path;
        qreal halfpw = pen.widthF() / 2;
        qreal xleft = halfpw, xright = m_size.width() - halfpw,
              ytop = halfpw, ybottom = m_size.height() - halfpw;

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

            painter->drawPath(path);

            QRectF rc(QPointF(0, 0), QPointF(halfpw, m_size.height()));
            painter->fillRect(rc, bDrawBg ? m_bgClr : innerBgclr);
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

            painter->drawPath(path);

            QRectF rc(QPointF(m_size.width(), 0), QPointF(m_size.width() - halfpw, m_size.height()));
            painter->fillRect(rc, bDrawBg ? m_bgClr : innerBgclr);
        }
    }
    else
    {
        QRectF rc(m_margin, m_margin, m_size.width(), m_size.height());
        painter->drawEllipse(rc);
    }
}