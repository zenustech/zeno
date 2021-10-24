#include "qdmgraphicsnode.h"
#include "qdmgraphicssocket.h"
#include <zeno/dop/Descriptor.h>

QDMGraphicsNode::QDMGraphicsNode()
{
    setFlag(QGraphicsItem::ItemIsMovable);
    setFlag(QGraphicsItem::ItemIsSelectable);

    label = new QGraphicsTextItem(this);
    label->setDefaultTextColor(QColor(0xcccccc));
    label->setPos(0, -SOCKSTRIDE);
}

QDMGraphicsNode::~QDMGraphicsNode()
{
    for (auto p: socketIns)
        delete p;
    for (auto p: socketOuts)
        delete p;
}

float QDMGraphicsNode::getHeight() const
{
    size_t count = std::max(socketIns.size(), socketOuts.size());
    return SOCKMARGINTOP + SOCKSTRIDE * count + SOCKMARGINBOT;
}

QRectF QDMGraphicsNode::boundingRect() const
{
    return QRectF(-QDMGraphicsSocket::SIZE, 1e-6f, WIDTH + QDMGraphicsSocket::SIZE * 2, getHeight() - 2e-6f);
}

void QDMGraphicsNode::paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget)
{
    if (isSelected()) {
        QPen pen;
        pen.setColor(QColor(0xff8800));
        pen.setWidthF(BORDER);
        painter->setPen(pen);
    } else {
        painter->setPen(Qt::NoPen);
    }
    painter->setBrush(QColor(0x555555));

    QPainterPath path;
    QRectF rect(0, 0, WIDTH, getHeight());
    path.addRoundedRect(rect, ROUND, ROUND);
    painter->drawPath(path.simplified());
}

QDMGraphicsSocketIn *QDMGraphicsNode::addSocketIn()
{
    auto socketIn = new QDMGraphicsSocketIn;
    socketIn->setParentItem(this);

    size_t index = socketIns.size();
    socketIn->setPos(-socketIn->SIZE / 2, SOCKMARGINTOP + SOCKSTRIDE * index);

    socketIns.push_back(socketIn);
    return socketIn;
}

QDMGraphicsSocketOut *QDMGraphicsNode::addSocketOut()
{
    auto socketOut = new QDMGraphicsSocketOut;
    socketOut->setParentItem(this);

    size_t index = socketOuts.size();
    socketOut->setPos(WIDTH + socketOut->SIZE / 2, SOCKMARGINTOP + SOCKSTRIDE * index);

    socketOuts.push_back(socketOut);
    return socketOut;
}

void QDMGraphicsNode::setupByName(QString name)
{
    setName(name);
    auto const &desc = zeno::dop::desc_of(name.toStdString());
    for (auto const &sockinfo: desc.inputs) {
        auto socket = addSocketIn();
        socket->setName(QString::fromStdString(sockinfo.name));
    }
    for (auto const &sockinfo: desc.outputs) {
        auto socket = addSocketOut();
        socket->setName(QString::fromStdString(sockinfo.name));
    }
}

void QDMGraphicsNode::setName(QString name)
{
    label->setPlainText(name);
}
