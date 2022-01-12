#include "../style/zenostyle.h"
#include "zcombobox.h"


ZComboBox::ZComboBox(QWidget *parent)
    : QComboBox(parent)
{
    init();
}

ZComboBox::~ZComboBox()
{

}

void ZComboBox::init()
{

}

QSize ZComboBox::sizeHint() const
{
    return ZenoStyle::dpiScaledSize(QSize(128, 25));
}

void ZComboBox::initStyleOption(ZStyleOptionComboBox* option)
{
    QStyleOptionComboBox opt;
    QComboBox::initStyleOption(&opt);
    *option = opt;

    option->bdrNormal = QColor(122, 122, 122);
    option->bdrHoverd = QColor(228, 228, 228);
    option->bdrSelected = QColor(122, 122, 122);

    //option->palette.setColor(QPalette::Active, QPalette::WindowText, QColor(228, 228, 228));
    //option->palette.setColor(QPalette::Inactive, QPalette::WindowText, QColor(158, 158, 158));

    option->clrBackground = QColor(50, 50, 50);
    option->clrBgHovered = QColor(50, 50, 50);
    option->clrText = QColor(229, 229, 229);

    option->btnNormal = QColor(50, 50, 50);
    option->btnHovered = QColor(50, 50, 50);
    option->btnHovered = QColor(50, 50, 50);

    option->textMargin = 5;
    option->palette.setColor(QPalette::ButtonText, option->clrText);
}

void ZComboBox::paintEvent(QPaintEvent* event)
{
    QStylePainter painter(this);
    painter.setPen(palette().color(QPalette::Text));
    // draw the combobox frame, focusrect and selected etc.
    ZStyleOptionComboBox opt;
    initStyleOption(&opt);
    painter.drawComplexControl(static_cast<QStyle::ComplexControl>(ZenoStyle::CC_ZenoComboBox), opt);
    painter.drawControl(static_cast<QStyle::ControlElement>(ZenoStyle::CE_ZenoComboBoxLabel), opt);
}
