#include "zenosocketitem.h"
#include "zgraphicstextitem.h"
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/viewparammodel.h>


ZenoSocketItem::ZenoSocketItem(
        const QPersistentModelIndex& viewSockIdx,
        bool bInput,
        const ImageElement& elem,
        const QSizeF& sz,
        QGraphicsItem* parent)
    : ZenoImageItem(elem, sz, parent)
    , m_bInput(bInput)
    , m_viewSockIdx(viewSockIdx)
    , m_status(STATUS_UNKNOWN)
    , m_svgHover(nullptr)
    , m_hoverSvg(elem.imageHovered)
    , m_noHoverSvg(elem.image)
    , sHorLargeMargin(ZenoStyle::dpiScaled(40))
    , sTopMargin(ZenoStyle::dpiScaled(10))
    , sHorSmallMargin(ZenoStyle::dpiScaled(5))
    , sBottomMargin(ZenoStyle::dpiScaled(10))
    , sLeftMargin(0)
    , sRightMargin(0)
{
    setCheckable(true);
    setSockStatus(STATUS_NOCONN);
    if (m_svg)
        m_svg->setPos(QPointF(sLeftMargin, sTopMargin));
}

int ZenoSocketItem::type() const
{
    return Type;
}

void ZenoSocketItem::setContentMargins(qreal left, qreal top, qreal right, qreal bottom)
{
    sTopMargin = top;
    sBottomMargin = bottom;
    sLeftMargin = left;
    sRightMargin = right;
    if (m_bInput) {
        sHorLargeMargin = left;
        sHorSmallMargin = right;
    } else {
        sHorLargeMargin = right;
        sHorSmallMargin = left;
    }
    m_svg->setPos(QPointF(sLeftMargin, sTopMargin));
}

void ZenoSocketItem::getContentMargins(qreal& left, qreal& top, qreal& right, qreal& bottom)
{
    left = sLeftMargin;
    top = sTopMargin;
    right = sRightMargin;
    bottom = sBottomMargin;
}

void ZenoSocketItem::setOffsetToName(const QPointF& offsetToName)
{
    m_offsetToName = offsetToName;
}

QPointF ZenoSocketItem::center() const
{
    QRectF rcImage = ZenoImageItem::boundingRect();
    QRectF br = boundingRect();
    QPointF cen = br.topLeft();
    if (m_bInput)
    {
        cen += QPointF(sHorLargeMargin + rcImage.width() / 2., sTopMargin + rcImage.height() / 2);
        QPointF c = mapToScene(cen);
        return c;
    }
    else
    {
        cen += QPointF(sHorSmallMargin + rcImage.width() / 2., sTopMargin + rcImage.height() / 2);
        QPointF c = mapToScene(cen);
        return c;
    }
}

QModelIndex ZenoSocketItem::paramIndex() const
{
    return m_viewSockIdx;
}

QRectF ZenoSocketItem::boundingRect() const
{
    QRectF rc = ZenoImageItem::boundingRect();
    if (m_bInput) {
        rc = rc.adjusted(-sHorLargeMargin, -sTopMargin, sHorSmallMargin, sBottomMargin);
    }
    else {
        rc = rc.adjusted(-sHorSmallMargin, -sTopMargin, sHorLargeMargin, sBottomMargin);
    }
    return QRectF(0, 0, rc.width(), rc.height());
    //return rc;
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
    switch (m_status)
    {
    case STATUS_CONNECTED:
        m_noHoverSvg = m_selected;
        m_hoverSvg = ":/icons/socket-on-hover.svg";
        delete m_svg;
        m_svg = new ZenoSvgItem(m_noHoverSvg, this);
        m_svg->setSize(m_size);
        m_svg->setPos(QPointF(sLeftMargin, sTopMargin));
        break;
    case STATUS_TRY_CONN:
        m_noHoverSvg = ":/icons/socket-on-hover.svg";
        m_hoverSvg = ":/icons/socket-on-hover.svg";
        delete m_svg;
        m_svg = new ZenoSvgItem(m_noHoverSvg, this);
        m_svg->setSize(m_size);
        m_svg->setPos(QPointF(sLeftMargin, sTopMargin));
        break;
    case STATUS_TRY_DISCONN:
    case STATUS_NOCONN:
        m_noHoverSvg = m_normal;
        m_hoverSvg = m_hoverSvg;
        delete m_svg;
        m_svg = new ZenoSvgItem(m_noHoverSvg, this);
        m_svg->setSize(m_size);
        m_svg->setPos(QPointF(sLeftMargin, sTopMargin));
        break;
    }
    update();
}

void ZenoSocketItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    delete m_svg;
    m_svg = new ZenoSvgItem(m_hoverSvg, this);
    m_svg->setSize(m_size);
    m_svg->setPos(QPointF(sLeftMargin, sTopMargin));
    QGraphicsObject::hoverEnterEvent(event);
}

void ZenoSocketItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    QGraphicsObject::hoverMoveEvent(event);
}

void ZenoSocketItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    delete m_svg;
    m_svg = new ZenoSvgItem(m_noHoverSvg, this);
    m_svg->setSize(m_size);
    m_svg->setPos(QPointF(sLeftMargin, sTopMargin));
    QGraphicsObject::hoverLeaveEvent(event);
}

void ZenoSocketItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    ZenoImageItem::mousePressEvent(event);
}

void ZenoSocketItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    ZenoImageItem::mouseReleaseEvent(event);
    m_svg->setPos(QPointF(sLeftMargin, sTopMargin));
    emit clicked(m_bInput);
}

void ZenoSocketItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    ZenoImageItem::paint(painter, option, widget);
}