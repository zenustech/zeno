#include "zstyleoption.h"


ZStyleOptionToolButton::ZStyleOptionToolButton()
    : buttonOpts(0)
    , roundCorner(0)
    , orientation(Qt::Horizontal)
    , hideText(false)
    , buttonEnabled(true)
    , bDown(false)
    , bgBrush(Qt::NoBrush)
    , bTextUnderIcon(false)
    , m_arrowOption(NO_ARROW)
    , m_iconOption(NO_ICON)
{
    this->type = Type;
}


ZStyleOptionComboBox::ZStyleOptionComboBox()
    : QStyleOptionComboBox()
    , textMargin(0)
{
}

ZStyleOptionComboBox::ZStyleOptionComboBox(const QStyleOptionComboBox &opt) : QStyleOptionComboBox(opt), textMargin(0) {
}


ZStyleOptionCheckBoxBar::ZStyleOptionCheckBoxBar() : QStyleOptionComplex(), bHovered(false) {

}

ZStyleOptionCheckBoxBar::ZStyleOptionCheckBoxBar(const QStyleOptionComplex &opt)
    : QStyleOptionComplex(opt), bHovered(false) {

}