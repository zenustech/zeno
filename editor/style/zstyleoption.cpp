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