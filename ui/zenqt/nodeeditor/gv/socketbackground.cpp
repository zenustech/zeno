#include "socketbackground.h"
#include "zenosocketitem.h"


SocketBackgroud::SocketBackgroud(bool bInput, bool bPanelLayout, QGraphicsItem* parent, Qt::WindowFlags wFlags)
    : ZLayoutBackground(parent, wFlags)
    , m_socket(nullptr)
    , m_bInput(bInput)
    , m_bPanelLayout(bPanelLayout)
{
    setColors(false, QColor(0, 0, 0, 0));
    connect(this, &SocketBackgroud::geometryChanged, this, &SocketBackgroud::onGeometryChanged);
}

void SocketBackgroud::setSocketItem(ZenoSocketItem* pSocket)
{
    m_socket = pSocket;
    m_socket->setParentItem(this);
}

void SocketBackgroud::onGeometryChanged()
{
    if (!m_socket)
        return;
    QRectF rc = this->boundingRect();
    QSizeF szSocket = m_socket->getSize();
    qreal x = 0, y = rc.center().y() - szSocket.height() / 2.;
    qreal w = szSocket.width();
    if (m_bInput) {
        x = -w / 2.;
    }
    else {
        x = rc.right() - m_socket->getSize().width() / 2;
    }

    if (m_bPanelLayout) {
        y = 0.;
    }

    QPointF newPos = QPointF(x, y);
    m_socket->setPos(newPos);
}