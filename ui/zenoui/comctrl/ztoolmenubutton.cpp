#include "ztoolmenubutton.h"
#include "../style/zstyleoption.h"
#include "../style/zenostyle.h"

CustomIconStyle::CustomIconStyle()
{
    mSize = 30;
}

CustomIconStyle::~CustomIconStyle()
{

}


void CustomIconStyle::SetCustomSize(int nSize)
{
    mSize = nSize;
}

int CustomIconStyle::pixelMetric(PixelMetric metric, const QStyleOption* option, const QWidget* widget) const
{
    int s = QCommonStyle::pixelMetric(metric, option, widget);
    if (metric == QStyle::PM_SmallIconSize) {
        s = mSize;
    }

    return s;

}


ZToolMenuButton::ZToolMenuButton(QWidget* parent)
    : ZToolButton(parent)
{
    setButtonOptions(ZToolButton::Opt_TextRightToIcon);
    //setArrowOption(ZStyleOptionToolButton::DOWNARROW);
    m_pMenu = new QMenu(this);
    m_pMenu->setProperty("cssClass", "menuButton");
    CustomIconStyle* pStyle = new CustomIconStyle;
    pStyle->SetCustomSize(ZenoStyle::dpiScaled(24));
    m_pMenu->setStyle(pStyle);
}

void ZToolMenuButton::addAction(const QString& action, const QString& icon)
{
    QAction* pAction = new QAction(QIcon(icon), action, this);
    connect(pAction, &QAction::triggered, this, [=]() {
        setText(action);
        this->setMinimumWidth(sizeHint().width());
        emit textChanged();
    });
    m_pMenu->addAction(pAction);
}

void ZToolMenuButton::mouseReleaseEvent(QMouseEvent* e) {
    //QSize size = ZToolButton::sizeHint();
    //if (e->x() >= (size.width() - ZenoStyle::dpiScaled(10)))
    //{
    //    QPoint pos;
    //    pos.setY(pos.y() + this->geometry().height());
    //    m_pMenu->exec(this->mapToGlobal(pos));
    //    return;
    //}
    emit clicked();
}


QSize ZToolMenuButton::sizeHint() const {
    QSize size = ZToolButton::sizeHint();
    //size.setWidth(size.width() + ZenoStyle::dpiScaled(12));
    return size;
}