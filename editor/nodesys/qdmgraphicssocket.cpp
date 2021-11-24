#include "qdmgraphicssocket.h"
#include "qdmgraphicsnode.h"
#include "qdmgraphicsscene.h"

ZENO_NAMESPACE_BEGIN

QDMGraphicsSocket::QDMGraphicsSocket()
    : label(new QGraphicsTextItem(this))
{
}

void QDMGraphicsSocket::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton) {
        auto parentScene = static_cast<QDMGraphicsScene *>(scene());
        parentScene->socketClicked(this);
    }

    QGraphicsItem::mousePressEvent(event);
}

QRectF QDMGraphicsSocket::boundingRect() const
{
    QRectF rect(-SIZE / 2, -SIZE / 2, SIZE, SIZE);
    //auto parentNode = static_cast<QDMGraphicsNode *>(parentItem());
    //return QRectF(0, 0, parentNode->boundingRect().width(), SIZE);
    return rect;
}

void QDMGraphicsSocket::paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget)
{
    QPainterPath pathContent;
    QRectF rect(-SIZE / 2, -SIZE / 2, SIZE, SIZE);
    pathContent.addRoundedRect(rect, ROUND, ROUND);
    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor(0xcccccc));
    painter->drawPath(pathContent.simplified());
}

void QDMGraphicsSocket::setName(std::string const &name)
{
    this->name = name;
    label->setPlainText(QString::fromStdString(name));
}

void QDMGraphicsSocket::setType(std::string const &type)
{
    this->type = type;
}

void QDMGraphicsSocket::setDefl(std::string const &defl)
{
    this->defl = defl;
}

ZENO_NAMESPACE_END
