#include "zicontoolbutton.h"


ZIconToolButton::ZIconToolButton(const QString& iconIdle, const QString& iconLight, QWidget* parent)
    : QToolButton(parent)
    , m_iconIdle(iconIdle)
    , m_iconLight(iconLight)
{
    QPixmap px(iconIdle);
    QIcon icon(iconIdle);
    QSize sz = px.size();
    setIconSize(sz);
    setIcon(icon);

    bool wtf = autoRaise();
    setAutoRaise(true);

    QString qss = "\
                QToolButton {\
                    border: 0px solid #8f8f91;\
                    border-radius: 0px;\
                    background-color: transparent;\
                    /*background-image: url(%1);*/\
                    padding: 0px;\
                }\
                \
                QToolButton:hover, QToolButton::pressed {\
                    /*background-image: url(%2);*/\
                }";
    QString stylesheet = qss.arg(iconIdle).arg(iconLight);
    //setStyleSheet(stylesheet);
}

ZIconToolButton::~ZIconToolButton()
{
}

QSize ZIconToolButton::sizeHint() const
{
    QSize sz = QToolButton::sizeHint();
    return sz;
}

void ZIconToolButton::enterEvent(QEvent* event)
{
    QToolButton::enterEvent(event);
    setIcon(QIcon(m_iconLight));
}

void ZIconToolButton::leaveEvent(QEvent* event)
{
    QToolButton::leaveEvent(event);
    setIcon(QIcon(m_iconIdle));
}

void ZIconToolButton::paintEvent(QPaintEvent* event)
{
    QToolButton::paintEvent(event);
}