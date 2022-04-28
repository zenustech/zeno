#ifndef __ZSHADOW_BUTTON_H__
#define __ZSHADOW_BUTTON_H__

#include <QPushButton>

class ZShadowButton : public QPushButton
{
    Q_OBJECT
public:
    explicit ZShadowButton(QWidget* parent = nullptr);
    explicit ZShadowButton(const QString& text, QWidget* parent = nullptr);
    ZShadowButton(const QIcon& icon, const QString& text, QWidget* parent = nullptr);
    void initStylesheet(
        qreal padding_left,
        qreal padding_top,
        qreal padding_right,
        qreal padding_bottom,
        qreal border_radius,
        qreal fontsize,
        bool bold);

private:
    void initShadow();
};

#endif