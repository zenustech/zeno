#include "zenosocketitem.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/model/modelrole.h>


ZenoSocketItem::ZenoSocketItem(const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent)
    : ZenoImageItem(elem, sz, parent)
    , m_bInput(false)
    , m_status(STATUS_UNKNOWN)
    , m_svgHover(nullptr)
    , m_hoverSvg(elem.imageHovered)
    , m_noHoverSvg(elem.image)
{
    setCheckable(true);
    setSockStatus(STATUS_NOCONN);
}

int ZenoSocketItem::type() const
{
    return Type;
}

void ZenoSocketItem::setOffsetToName(const QPointF& offsetToName)
{
    m_offsetToName = offsetToName;
}

void ZenoSocketItem::socketNamePosition(const QPointF& nameScenePos)
{
    QPointF namePos = mapFromScene(nameScenePos);
    setPos(namePos + m_offsetToName);
}

QRectF ZenoSocketItem::boundingRect() const
{
    static int sLargeMargin = ZenoStyle::dpiScaled(20);
    static int sSmallMargin = ZenoStyle::dpiScaled(10);

    QRectF rc = ZenoImageItem::boundingRect();
    if (m_bInput) {
        rc = rc.adjusted(-sLargeMargin, -sSmallMargin, sLargeMargin, sSmallMargin);
    }
    else {
        rc = rc.adjusted(-sLargeMargin, -sSmallMargin, sLargeMargin, sSmallMargin);
    }
    return rc;
}

void ZenoSocketItem::setSocketInfo(QPersistentModelIndex index, bool input, SOCKET_INFO info)
{
    m_index = index;
    m_bInput = input;
    m_info = info;
}

void ZenoSocketItem::setSockStatus(SOCK_STATUS status)
{
    if (m_status == status)
        return;

    if (status == STATUS_NOCONN || status == STATUS_TRY_DISCONN)
    {
        if (m_bInput) {
            INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
            if (inputs.find(m_info.name) != inputs.end()) {
                if (!inputs[m_info.name].linkIndice.isEmpty()) {
                    status = STATUS_CONNECTED;
                }
            }
        } else {
            OUTPUT_SOCKETS outputs = m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
            if (outputs.find(m_info.name) != outputs.end()) {
                if (!outputs[m_info.name].linkIndice.isEmpty()) {
                    status = STATUS_CONNECTED;
                }
            }
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
        break;
    case STATUS_TRY_CONN:
        m_noHoverSvg = ":/icons/socket-on-hover.svg";
        m_hoverSvg = ":/icons/socket-on-hover.svg";
        delete m_svg;
        m_svg = new ZenoSvgItem(m_noHoverSvg, this);
        m_svg->setSize(m_size);
        break;
    case STATUS_TRY_DISCONN:
    case STATUS_NOCONN:
        m_noHoverSvg = m_normal;
        m_hoverSvg = m_hoverSvg;
        delete m_svg;
        m_svg = new ZenoSvgItem(m_noHoverSvg, this);
        m_svg->setSize(m_size);
        break;
    }
    update();
}

void ZenoSocketItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    delete m_svg;
    m_svg = new ZenoSvgItem(m_hoverSvg, this);
    m_svg->setSize(m_size);
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
    QGraphicsObject::hoverLeaveEvent(event);
}

void ZenoSocketItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    ZenoImageItem::paint(painter, option, widget);
}