#include "qdmgraphicssocketout.h"
#include <QTextDocument>

QDMGraphicsSocketOut::QDMGraphicsSocketOut()
{
    label->setPos(-SIZE * 10 - SIZE / 2, -SIZE * 2 / 3);
    label->setTextWidth(SIZE * 10);
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
