#include "qdmgraphicsnode.h"
#include <zeno/dop/Descriptor.h>

QDMGraphicsNode::QDMGraphicsNode()
{
    setFlag(QGraphicsItem::ItemIsMovable);
    setFlag(QGraphicsItem::ItemIsSelectable);
}

QDMGraphicsNode::~QDMGraphicsNode()
{
    for (auto p: socketIns)
        delete p;
    for (auto p: socketOuts)
        delete p;
}

QRectF QDMGraphicsNode::boundingRect() const
{
    size_t count = std::max(socketIns.size(), socketOuts.size());
    auto node_height = SOCKMARGINTOP + SOCKSTRIDE * count + SOCKMARGINBOT;
    return QRectF(0, 0, WIDTH, node_height);
}

void QDMGraphicsNode::paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget)
{
    QPainterPath pathContent;
    QRectF rect = boundingRect();
    pathContent.addRoundedRect(rect, ROUND, ROUND);
    if (isSelected()) {
        QPen pen;
        pen.setColor(Qt::blue);
        pen.setWidthF(BORDER);
        painter->setPen(pen);
    } else {
        painter->setPen(Qt::NoPen);
    }
    painter->setBrush(Qt::red);
    painter->drawPath(pathContent.simplified());
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
