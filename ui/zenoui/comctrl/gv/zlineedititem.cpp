#include "zlineedititem.h"
#include <zenoedit/zenoapplication.h>


ZLineEditItem::ZLineEditItem(QGraphicsItem* parent)
    : _base(parent)
{
    setTextInteractionFlags(Qt::TextEditorInteraction);
    setBackground(QColor(28, 28, 28));

    QColor clr = QColor(255, 255, 255);
    clr.setAlphaF(0.4);
    setDefaultTextColor(clr);

    QFont font = zenoApp->font();
    font.setPointSize(10);
    setFont(font);
    setMargins(15, 8, 15, 8);
}

ZLineEditItem::ZLineEditItem(const QString& text, QGraphicsItem* parent)
    : _base(text, QFont(), QColor(), parent)
{
    setTextInteractionFlags(Qt::TextEditorInteraction);
    setBackground(QColor(28, 28, 28));

    QColor clr = QColor(255, 255, 255);
    clr.setAlphaF(0.4);
    setDefaultTextColor(clr);


    QFont font = zenoApp->font();
    font.setPointSize(10);    setFont(font);
    setMargins(15, 8, 15, 8);
}

ZLineEditItem::~ZLineEditItem()
{

}

void ZLineEditItem::setText(const QString& text)
{
    _base::setText(text);
}

QRectF ZLineEditItem::boundingRect() const
{
    QRectF rc = _base::boundingRect();
    return rc;
}
