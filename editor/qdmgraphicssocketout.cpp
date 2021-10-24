#include "qdmgraphicssocketout.h"
#include <QTextDocument>
#include "qdmgraphicsnode.h"

QDMGraphicsSocketOut::QDMGraphicsSocketOut()
{
    label->setPos(-QDMGraphicsNode::WIDTH - SIZE / 2, -SIZE * 0.7f);
    label->setTextWidth(QDMGraphicsNode::WIDTH);
    auto doc = label->document();
    auto opt = doc->defaultTextOption();
    opt.setAlignment(Qt::AlignRight);
    doc->setDefaultTextOption(opt);
}

void QDMGraphicsSocketOut::paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget)
{
    QPainterPath pathContent;
    QRectF rect(-SIZE / 2, -SIZE / 2, SIZE, SIZE);
    pathContent.addRoundedRect(rect, ROUND, ROUND);
    painter->setPen(Qt::NoPen);
    painter->setBrush(Qt::blue);
    painter->drawPath(pathContent.simplified());
}
