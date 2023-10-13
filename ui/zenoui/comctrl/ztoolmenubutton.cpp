#include "ztoolmenubutton.h"
#include "../style/zstyleoption.h"
#include "../style/zenostyle.h"

ZToolMenuButton::ZToolMenuButton(QWidget* parent)
    : ZToolButton(parent)
{
    setButtonOptions(ZToolButton::Opt_TextRightToIcon);
    setArrowOption(ZStyleOptionToolButton::DOWNARROW);
    setBorderColor(QColor(255,255,255,102));
    m_pMenu = new QMenu(this);
    m_pMenu->setProperty("cssClass", "menuButton");
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
    QSize size = ZToolButton::sizeHint();
    if (e->x() >= (size.width() - ZenoStyle::dpiScaled(10)))
    {
        QPoint pos;
        pos.setY(pos.y() + this->geometry().height());
        m_pMenu->exec(this->mapToGlobal(pos));
        return;
    }
    emit clicked();
}


QSize ZToolMenuButton::sizeHint() const {
    QSize size = ZToolButton::sizeHint();
    size.setWidth(size.width() + ZenoStyle::dpiScaled(12));
    return size;
}