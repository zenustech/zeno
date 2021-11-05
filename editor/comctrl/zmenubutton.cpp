#include "zmenubutton.h"

#include "../style/zenostyle.h"
#include "../style/zstyleoption.h"


ZMenuButton::ZMenuButton(ButtonOption option, const QIcon& icon, const QSize& iconSize, const QString& text, QWidget* parent)
    : ZToolButton(option, icon, iconSize, text, parent)
{
    setMouseTracking(true);
    connect(this, SIGNAL(clicked()), this, SIGNAL(popup()));
    connect(this, SIGNAL(popup()), this, SLOT(popupChildWidget()));
}

ZMenuButton::~ZMenuButton()
{
}

void ZMenuButton::popupChildWidget()
{
    //todo
    //ZPopupWidget popup(this);
    //popup.exec(pGlobal.x(), pGlobal.y() + height() + margin, nWidth, nHeight);
}

void ZMenuButton::initStyleOption(ZStyleOptionToolButton* option) const
{
    ZToolButton::initStyleOption(option);
    option->features |= QStyleOptionToolButton::Menu;
}

void ZMenuButton::paintEvent(QPaintEvent* event)
{
    ZToolButton::paintEvent(event);
}

bool ZMenuButton::event(QEvent* e)
{
    return ZToolButton::event(e);
}