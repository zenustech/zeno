#include "zenoparamnameitem.h"


ZenoParamNameItem::ZenoParamNameItem(const QString &paramName, QGraphicsLayoutItem *parent, bool isLayout)
    : QGraphicsLayoutItem(parent, isLayout)
{
    QGraphicsTextItem *pItem = new QGraphicsTextItem(paramName);
    setGraphicsItem(pItem);
}

QSizeF ZenoParamNameItem::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
    switch (which) {
        case Qt::MinimumSize:
            return QSizeF(32, 32);
        case Qt::PreferredSize:
            return QSizeF(160, 32);
        case Qt::MaximumSize:
            return QSizeF(1000, 32);
        default:
            return QSizeF(300, 32);
    }
}