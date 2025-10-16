#include "selfdefinedqmltype.h"

SelfDefinedQMLType::SelfDefinedQMLType()
{
    setFlags(QQuickItem::ItemHasContents);
    QSGSimNode = NULL;
    qDebug() << "SelfDefinedQMLType::SelfDefinedQMLType() was called!";
}

SelfDefinedQMLType::~SelfDefinedQMLType()
{
    qDebug() << "SelfDefinedQMLType::~SelfDefinedQMLType start";
    if (QSGSimNode)
    {
        //Must comment the following, otherwise, there will be an error!
        //seems that the qt can handle resource itself.

        //delete QSGSimNode;
    }
    qDebug() << "SelfDefinedQMLType::~SelfDefinedQMLType end";
}

void SelfDefinedQMLType::changeColor()
{
    if (!QSGSimNode) {
        if (QColor(255, 0, 0, 127) == QSGSimNode->color())
            QSGSimNode->setColor(QColor(255, 0, 0, 127));
        else
            QSGSimNode->setColor(QColor(0, 255, 0, 127));
    }
}

QSGNode* SelfDefinedQMLType::updatePaintNode(QSGNode* node, UpdatePaintNodeData*)
{
    QSGSimNode = static_cast<QSGSimpleRectNode*>(node);
    if (!QSGSimNode) {
        QSGSimNode = new QSGSimpleRectNode();
        QColor myColor = QColor(255, 0, 0, 127);
        QSGSimNode->setColor(myColor);
        QSGSimNode->setRect(10, 10, 400, 400);
    }
    return QSGSimNode;
}

void SelfDefinedQMLType::keyPressEvent(QKeyEvent* event)
{
    if (Qt::Key_Left == event->key())
        changeColor();
}