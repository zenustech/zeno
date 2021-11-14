#include "qdmgraphicsnode.h"
#include "qdmgraphicssocket.h"
#include "qdmgraphicsscene.h"
#include <zeno/dop/Descriptor.h>
#include <zeno/ztd/memory.h>
#include <zeno/ztd/assert.h>
#include <QLineEdit>
#include <algorithm>

ZENO_NAMESPACE_BEGIN

QDMGraphicsNode::QDMGraphicsNode()
{
    setFlag(QGraphicsItem::ItemIsMovable);
    setFlag(QGraphicsItem::ItemIsSelectable);

    label = std::make_unique<QGraphicsTextItem>(this);
    label->setDefaultTextColor(QColor(0xcccccc));
    label->setPos(0, -SOCKSTRIDE);
}

QDMGraphicsNode::~QDMGraphicsNode() = default;

dop::Node *QDMGraphicsNode::getDopNode() const
{
    return dopNode.get();
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

    socketIns.emplace_back(socketIn);
    return socketIn;
}

QDMGraphicsSocketOut *QDMGraphicsNode::addSocketOut()
{
    auto socketOut = new QDMGraphicsSocketOut;
    socketOut->setParentItem(this);

    size_t index = socketOuts.size();
    socketOut->setPos(WIDTH + socketOut->SIZE / 2, SOCKMARGINTOP + SOCKSTRIDE * index);

    socketOuts.emplace_back(socketOut);
    return socketOut;
}

void QDMGraphicsNode::initAsSubnet()
{
    initByType("CustomSubnet");
}

void QDMGraphicsNode::initByType(QString type)
{
    auto const &desc = dop::descriptor_table().at(type.toStdString());
    dopNode = desc.create();
    for (auto const &sockinfo: desc.inputs) {
        auto socket = addSocketIn();
        socket->setName(QString::fromStdString(sockinfo.name));
    }
    for (auto const &sockinfo: desc.outputs) {
        auto socket = addSocketOut();
        socket->setName(QString::fromStdString(sockinfo.name));
    }

    auto name = getScene()->allocateNodeName(type.toStdString());
    setName(QString::fromStdString(name));
}

void QDMGraphicsNode::setName(QString name)
{
    label->setPlainText(name);
    dopNode->name = name.toStdString();
}

std::string const &QDMGraphicsNode::getType()
{
    return dopNode->desc->name;
}

std::string const &QDMGraphicsNode::getName()
{
    return dopNode->name;
}

QDMGraphicsSocketIn *QDMGraphicsNode::socketInAt(size_t index)
{
    return socketIns.at(index).get();
}

QDMGraphicsSocketOut *QDMGraphicsNode::socketOutAt(size_t index)
{
    return socketOuts.at(index).get();
}

QDMGraphicsScene *QDMGraphicsNode::getScene() const
{
    return static_cast<QDMGraphicsScene *>(scene());
}

void QDMGraphicsNode::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::RightButton) {
        getScene()->removeNode(this);
        return;
    }

    QGraphicsItem::mousePressEvent(event);
}

void QDMGraphicsNode::unlinkAll()
{
    for (auto const &p: socketIns) {
        p->unlinkAll();
    }
    for (auto const &p: socketOuts) {
        p->unlinkAll();
    }
}

size_t QDMGraphicsNode::socketInIndex(QDMGraphicsSocketIn *socket)
{
    auto it = find(begin(socketIns), end(socketIns), ztd::stale_ptr(socket));
    ZENO_ASSERT(it != end(socketIns));
    return it - begin(socketIns);
}

size_t QDMGraphicsNode::socketOutIndex(QDMGraphicsSocketOut *socket)
{
    auto it = find(begin(socketOuts), end(socketOuts), ztd::stale_ptr(socket));
    ZENO_ASSERT(it != end(socketOuts));
    return it - begin(socketOuts);
}

void QDMGraphicsNode::socketUnlinked(QDMGraphicsSocketIn *socket)
{
    auto &sockIn = dopNode->inputs.at(socketInIndex(socket));
    sockIn.node = nullptr;
    sockIn.sockid = 0;
}

void QDMGraphicsNode::socketLinked(QDMGraphicsSocketIn *socket, QDMGraphicsSocketOut *srcSocket)
{
    auto srcNode = static_cast<QDMGraphicsNode *>(srcSocket->parentItem());
    auto &sockIn = dopNode->inputs.at(socketInIndex(socket));
    sockIn.node = srcNode->dopNode.get();
    sockIn.sockid = srcNode->socketOutIndex(srcSocket);
}

void QDMGraphicsNode::socketValueChanged(QDMGraphicsSocketIn *socket)
{
    auto &sockIn = dopNode->inputs.at(socketInIndex(socket));
    sockIn.value = socket->value;
}

ZENO_NAMESPACE_END
