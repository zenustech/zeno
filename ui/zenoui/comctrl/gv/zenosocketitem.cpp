#include "zenosocketitem.h"
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/modelrole.h>


ZenoSocketItem::ZenoSocketItem(
        const QString& sockName,
        bool bInput,
        QPersistentModelIndex nodeIdx,
        const ImageElement& elem,
        const QSizeF& sz,
        QGraphicsItem* parent)
    : ZenoImageItem(elem, sz, parent)
    , m_bInput(bInput)
    , m_name(sockName)
    , m_index(nodeIdx)
    , m_status(STATUS_UNKNOWN)
    , m_svgHover(nullptr)
    , m_hoverSvg(elem.imageHovered)
    , m_noHoverSvg(elem.image)
    , sHorLargeMargin(ZenoStyle::dpiScaled(40))
    , sTopMargin(ZenoStyle::dpiScaled(10))
    , sHorSmallMargin(ZenoStyle::dpiScaled(5))
    , sBottomMargin(ZenoStyle::dpiScaled(10))
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

QRectF ZenoSocketItem::boundingRect() const
{
    QRectF rc = ZenoImageItem::boundingRect();
    if (m_bInput) {
        rc = rc.adjusted(-sHorLargeMargin, -sTopMargin, sHorSmallMargin, sBottomMargin);
    }
    else {
        rc = rc.adjusted(-sHorSmallMargin, -sTopMargin, sHorLargeMargin, sBottomMargin);
    }
    return rc;
}

void ZenoSocketItem::updateSockName(const QString& sockName)
{
    m_name = sockName;
}

bool ZenoSocketItem::getSocketInfo(bool& bInput, QString& nodeid, QString& sockName)
{
    Q_ASSERT(m_index.isValid());
    if (!m_index.isValid())
        return false;

    bInput = m_bInput;
    nodeid = m_index.data(ROLE_OBJID).toString();
    sockName = m_name;
    return true;
}

void ZenoSocketItem::setSockStatus(SOCK_STATUS status)
{
    if (m_status == status)
        return;

    if (status == STATUS_NOCONN || status == STATUS_TRY_DISCONN)
    {
        if (m_bInput) {
            INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
            if (inputs.find(m_name) != inputs.end()) {
                if (!inputs[m_name].linkIndice.isEmpty()) {
                    status = STATUS_CONNECTED;
                }
            }
        } else {
            OUTPUT_SOCKETS outputs = m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
            if (outputs.find(m_name) != outputs.end()) {
                if (!outputs[m_name].linkIndice.isEmpty()) {
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

void ZenoSocketItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    ZenoImageItem::mousePressEvent(event);
}

void ZenoSocketItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    ZenoImageItem::mouseReleaseEvent(event);
    emit clicked(m_bInput);
}

void ZenoSocketItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    ZenoImageItem::paint(painter, option, widget);
}