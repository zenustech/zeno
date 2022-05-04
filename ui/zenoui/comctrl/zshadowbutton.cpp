#include "zshadowbutton.h"
#include <zenoui/comctrl/effect/innershadoweffect.h>


ZShadowButton::ZShadowButton(QWidget* parent)
    : QPushButton(parent)
{
    initStylesheet(10, 2, 10, 2, 4, 12, false);
    initShadow();
}

ZShadowButton::ZShadowButton(const QString &text, QWidget *parent)
    : QPushButton(text, parent)
{
    initStylesheet(10, 2, 10, 2, 4, 12, false);
    initShadow();
}

ZShadowButton::ZShadowButton(const QIcon &icon, const QString &text, QWidget *parent)
    : QPushButton(icon, text, parent)
{
    initStylesheet(10, 2, 10, 2, 4, 12, false);
    initShadow();
}

void ZShadowButton::initStylesheet(qreal padding_left, qreal padding_top, qreal padding_right, qreal padding_bottom,
                                   qreal border_radius, qreal fontsize, bool bold)
{
    QString qssTemplate = 
        "QPushButton\
        {\
            padding-top: %1px;\
            padding-left: %2px;\
            padding-right: %3px;\
            padding-bottom: %4px;\
            border: 0.5px solid rgba(0,0,0,1);\
            border-radius: 3px;\
            background-color: #2B2B2B;\
            font: %5 %6px 'HarmonyOS Sans';\
            color: #8A8A8C;\
            background-position: center;\
	        background-repeat: no-repeat;\
        }\
\
        QPushButton:pressed, QPushButton:checked {\
            background-color: rgb(26, 26, 26);\
	        background-image: url(:/icons/ic_handle_free-on.svg);\
        }";
    QString qss = qssTemplate.arg(QString::number(padding_top))
                      .arg(QString::number(padding_left))
                      .arg(QString::number(padding_right))
                      .arg(QString::number(padding_bottom))
                      //.arg(QString::number(border_radius))
                      .arg(bold ? "bold" : "")
                      .arg(QString::number(fontsize));
    setStyleSheet(qss);
}

void ZShadowButton::initShadow()
{
    InnerShadowEffect *effect = new InnerShadowEffect;
    setGraphicsEffect(effect);
}