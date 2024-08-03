#include "zenosocketitem.h"
#include "zgraphicstextitem.h"
#include "zgraphicsnetlabel.h"
#include "style/zenostyle.h"
#include "uicommon.h"
#include "zassert.h"
#include "widgets/ztooltip.h"
#include "util/uihelper.h"

#define BASED_ON_SPEHERE


ZenoSocketItem::ZenoSocketItem(
        const QPersistentModelIndex& viewSockIdx,
        const QSizeF& sz,
        bool bInnerSocket,
        QGraphicsItem* parent
)
    : _base(parent)
    , m_paramIdx(viewSockIdx)
    , m_status(STATUS_UNKNOWN)
    , m_size(sz)
    , m_bHovered(false)
    , m_bInnerSocket(bInnerSocket)
    , m_innerSockMargin(0)
    , m_socketXOffset(0)
{
    m_bInput = m_paramIdx.data(ROLE_ISINPUT).toBool();
#ifndef BASED_ON_SPEHERE
    m_innerSockMargin = ZenoStyle::dpiScaled(15);
    m_socketXOffset = ZenoStyle::dpiScaled(24);
#endif
    if (!m_bInnerSocket)
    {
        setData(GVKEY_SIZEHINT, m_size);
        setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    }

    QColor sockClr = m_paramIdx.data(ROLE_PARAM_SOCKET_CLR).value<QColor>();
    setBrush(sockClr, sockClr);
    //if (m_paramIdx.data(ROLE_SOCKET_TYPE).toInt() == zeno::Socket_WildCard)
    //    setBrush(QColor("#4B9EF4"), QColor("#5FD2FF"));
    //else
    //    setBrush(QColor("#CCA44E"), QColor("#FFF000"));
    setSockStatus(STATUS_NOCONN);
    setAcceptHoverEvents(true);
}

ZenoSocketItem::~ZenoSocketItem()
{
}

int ZenoSocketItem::type() const
{
    return Type;
}

QPointF ZenoSocketItem::center() const
{
    if (m_bInnerSocket) {
        return this->sceneBoundingRect().center();
    }
    else
    {
        //(0, 0) is the position of socket.
        QPointF center = mapToScene(QPointF(m_size.width() / 2., m_size.height() / 2.));
        return center;
    }
}

QSizeF ZenoSocketItem::getSize() const {
    return m_size;
}

void ZenoSocketItem::setBrush(const QBrush& brush, const QBrush& brushOn)
{
    m_brush = brush;
    m_brushOn = brushOn;
}

void ZenoSocketItem::setInnerKey(const QString& key)
{
    m_innerKey = key;
}

QString ZenoSocketItem::innerKey() const
{
    return m_innerKey;
}

QModelIndex ZenoSocketItem::paramIndex() const
{
    return m_paramIdx;
}

QRectF ZenoSocketItem::boundingRect() const
{
    if (m_bInnerSocket)
    {
        QRectF rc(QPointF(0, 0), m_size + QSize(2 * m_innerSockMargin, 2 * m_innerSockMargin));
        return rc;
    }
    else
    {
        QSizeF wholeSize = QSizeF(m_size.width() + m_socketXOffset, m_size.height());
        if (m_bInput)
            return QRectF(QPointF(-m_socketXOffset, 0), wholeSize);
        else
            return QRectF(QPointF(0, 0), wholeSize);
    }
}

bool ZenoSocketItem::isInputSocket() const
{
    return m_bInput;
}

QString ZenoSocketItem::nodeIdent() const
{
    return m_paramIdx.isValid() ? m_paramIdx.data(ROLE_NODE_NAME).toString() : "";
}

void ZenoSocketItem::setHovered(bool bHovered)
{
    m_bHovered = bHovered;
    if (m_bHovered && m_paramIdx.isValid())
    {
        QString name = m_paramIdx.data(ROLE_PARAM_NAME).toString();
        QString type = UiHelper::getTypeDesc(m_paramIdx.data(ROLE_PARAM_TYPE).value<zeno::ParamType>());
        ZToolTip::showText(QCursor::pos() + QPoint(ZenoStyle::dpiScaled(10), 0), name + " ( " + (type.isEmpty() ? "null" : type) + " )");
    }
    else
    {
        ZToolTip::hideText();
    }

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
    {
        Qt::KeyboardModifiers modifiers = event->modifiers();
        if (Qt::LeftButton == event->button()) {
            emit clicked(m_bInput);
        }
    }
}

QVariant ZenoSocketItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
    if (change == QGraphicsItem::ItemVisibleHasChanged)
    {
        if (m_bHovered && !this->isVisible())
            m_bHovered = false;
    }
    return value;
}

QString ZenoSocketItem::netLabel() const
{
    return "";
}

void ZenoSocketItem::onCustomParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (!roles.empty() && roles[0] == ROLE_PARAM_SOCKET_CLR) {
        if (m_paramIdx.data(ROLE_PARAM_NAME).toString() == topLeft.data(ROLE_PARAM_NAME).toString()) {
            m_brushOn = topLeft.data(ROLE_PARAM_SOCKET_CLR).value<QColor>();
            m_brush = topLeft.data(ROLE_PARAM_SOCKET_CLR).value<QColor>();
            update();
        }
    }
}

void ZenoSocketItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
#if 1
    if (editor_factor < 0.2)
        return;
#endif
    painter->setRenderHint(QPainter::Antialiasing, true);

#ifdef BASED_ON_SPEHERE
    QRectF rc(m_innerSockMargin, m_innerSockMargin, m_size.width(), m_size.height());
    bool bOn = m_status == STATUS_TRY_CONN || m_status == STATUS_CONNECTED || m_bHovered;
    painter->setPen(Qt::NoPen);
    painter->setBrush(bOn ? m_brushOn : m_brush);
    painter->drawEllipse(rc);
#else
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

    if (!m_bInnerSocket)
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
    else
    {
        QRectF rc(m_innerSockMargin, m_innerSockMargin, m_size.width(), m_size.height());
        painter->drawEllipse(rc);
    }
#endif
}

ZenoObjSocketItem::ZenoObjSocketItem(const QPersistentModelIndex& viewSockIdx, const QSizeF& sz, bool bInnerSocket, QGraphicsItem* parent)
    : _base(viewSockIdx, sz, bInnerSocket, parent)
    , m_bInput(true)
{
    m_bInput = viewSockIdx.data(ROLE_ISINPUT).toBool();
}

ZenoObjSocketItem::~ZenoObjSocketItem()
{
}

void ZenoObjSocketItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    //object socket
    auto paramIdx = paramIndex();
    QRectF rc = boundingRect();
    auto status = sockStatus();
    bool bOn = status == STATUS_TRY_CONN || status == STATUS_CONNECTED || m_bHovered;
    //TODO: 这些其实可以缓存，只有子图才会修改
    PARAM_LINKS links = paramIdx.data(ROLE_LINKS).value<PARAM_LINKS>();
    zeno::SocketType type = static_cast<zeno::SocketType>(paramIdx.data(ROLE_SOCKET_TYPE).toInt());
    const QString& name = paramIdx.data(ROLE_PARAM_NAME).toString();

    qreal xLeft = rc.left();
    qreal xRight = rc.right();
    qreal ytop = rc.top();
    qreal ybottom = rc.bottom();

    painter->setRenderHint(QPainter::Antialiasing, true);

    if (type == zeno::Socket_Owning)
    {
        ZASSERT_EXIT(m_bInput);

        QPen pen(QColor("#4B9EF4"), 4);
        pen.setJoinStyle(Qt::MiterJoin);
        painter->setPen(pen);

        bool bOwn = !links.isEmpty();
        if (bOwn) {
            painter->setBrush(QColor("#4B9EF4"));
        }

        QPainterPath path;
        path.moveTo(xLeft, ytop);
        qreal yMiddle = rc.height() * 0.6;
        qreal radius = rc.height() - yMiddle;
        path.lineTo(0, yMiddle);
        path.arcTo(QRectF(QPointF(xLeft, ybottom - 2 * radius), QPointF(xLeft + 2 * radius, ybottom)), 180, 90);
        path.lineTo(xRight - radius, ybottom);
        path.arcTo(QRectF(QPointF(xRight - 2 * radius, ytop), QPointF(xRight, ybottom)), 270, 90);
        path.lineTo(xRight, ytop);

        if (bOwn) {
            path.lineTo(xLeft, ytop);
        }

        painter->drawPath(path);

        pen = QPen(Qt::white, 2);
        painter->setPen(pen);
        rc.adjust(0, -3, 0, -3);

        painter->drawText(rc, Qt::AlignHCenter | Qt::AlignTop, name);
    }
    else if (type == zeno::Socket_ReadOnly)
    {
        ZASSERT_EXIT(m_bInput);
        QPen pen(QColor("#4B9EF4"), 4);
        pen.setJoinStyle(Qt::MiterJoin);
        painter->setPen(pen);

        QPainterPath path;
        path.moveTo(xLeft, ybottom);
        path.lineTo(xLeft, ytop);
        path.lineTo(xRight, ytop);
        path.lineTo(xRight, ybottom);
        painter->drawPath(path);

        pen = QPen(Qt::white, 2);
        
        painter->setPen(pen);
        rc.adjust(0, 2, 0, 2);
        painter->drawText(rc, Qt::AlignCenter, name);
    }
    else if (type == zeno::Socket_Clone)
    {
        ZASSERT_EXIT(m_bInput);
        QPen pen(QColor("#4B9EF4"), 4);
        QBrush brush(QColor("#4B9EF4"));
        painter->setPen(pen);
        painter->setBrush(brush);

        qreal radius = rc.height() * 0.2;
        painter->drawRoundedRect(rc, radius, radius);

        pen = QPen(Qt::white, 2);
        painter->setPen(pen);
        painter->drawText(rc, Qt::AlignCenter, name);
    }
    else {
        ZASSERT_EXIT(!m_bInput && type == zeno::Socket_Output);
        qreal radius = rc.height() * 0.2;
        QPen pen = QPen(Qt::white, 2);

        //观察是否被own了
        if (!links.empty()) {
            QModelIndex targetSocket = links[0].data(ROLE_INSOCK_IDX).toModelIndex();
            ZASSERT_EXIT(targetSocket.isValid());
            zeno::SocketType sockType = static_cast<zeno::SocketType>(targetSocket.data(ROLE_SOCKET_TYPE).toInt());
            if (sockType == zeno::Socket_Owning) {
                painter->fillRect(rc, QColor("#AAAAAA"));
                pen = QPen(QColor("#777777"), 2);
            }
            else {
                painter->fillRect(rc, QColor("#4B9EF4"));
            }
        }
        else {
            painter->fillRect(rc, QColor("#4B9EF4"));
        }

        painter->setPen(pen);
        painter->drawText(rc, Qt::AlignCenter, name);
    }
}
