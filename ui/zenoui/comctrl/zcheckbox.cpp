#include "zcheckbox.h"


ZCheckBox::ZCheckBox(QWidget* parent)
    : QCheckBox(parent)
{

}

void ZCheckBox::paintEvent(QPaintEvent* event)
{
    QStylePainter p(this);
    QStyleOptionButton opt;
    initStyleOption(&opt);
    p.drawControl(QStyle::CE_CheckBox, opt);
}