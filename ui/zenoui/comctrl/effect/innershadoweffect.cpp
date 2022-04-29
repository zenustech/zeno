#include "innershadoweffect.h"
#include <QPainter>
#include <QPushButton>
#include <QtWidgets/private/qgraphicseffect_p.h>


InnerShadowEffect::InnerShadowEffect(QObject *parent)
    : QGraphicsEffect(parent)
{

}

InnerShadowEffect::~InnerShadowEffect()
{

}

QRectF InnerShadowEffect::boundingRectFor(const QRectF &sourceRect) const
{
    return QGraphicsEffect::boundingRectFor(sourceRect);
}

void InnerShadowEffect::draw(QPainter *painter)
{
    QPoint offset;
    const QPixmap pixmap = sourcePixmap(Qt::DeviceCoordinates, &offset, NoPad);
    if (pixmap.isNull())
        return;

    QSize sz = pixmap.size();
    qreal w = sz.width(), h = sz.height();
    painter->drawPixmap(QPointF(0, 0), pixmap);

    qreal left_padding = 4, vpadding = 1;

    QWidget* hostWid = const_cast<QWidget*>(this->source()->widget());

    QColor clrUpperShadow, clrDownShadow;

    clrUpperShadow = QColor(54, 54, 54);
    clrDownShadow = QColor(34, 34, 34);

    if (QPushButton* pBtn = qobject_cast<QPushButton*>(hostWid))
    {
        if (pBtn->isChecked() || pBtn->isDown())
        {
            clrUpperShadow = QColor(36, 35, 35);
            clrDownShadow = QColor(19, 19, 19);
        }
    }

    painter->setPen(QPen(clrUpperShadow, 1));
    painter->drawLine(left_padding, vpadding, sz.width() - 1 * left_padding, vpadding);
    //painter->drawArc(1, 1, 1, 1, 90 * 16, 180 * 16);
    //painter->drawArc(w - 3, 1, 1, 1, 0 * 16, 90 * 16);

    painter->setPen(QPen(clrDownShadow, 1));
    painter->drawLine(left_padding, sz.height() - vpadding - 1, sz.width(), sz.height() - vpadding - 1);
}

void InnerShadowEffect::sourceChanged(ChangeFlags flags)
{
    QGraphicsEffect::sourceChanged(flags);
}