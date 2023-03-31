#include "zcheckbox.h"
#include <QSvgRenderer>


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
    if (checkState() == Qt::Checked) {
        QSvgRenderer svgRnder(QString(":/icons/checkbox-light.svg"));
        svgRnder.render(&p, rect());
    }
}