#ifndef __ZOBJECTBUTTON_H__
#define __ZOBJECTBUTTON_H__

#include "../comctrl/ztoolbutton.h"

class ZObjectButton : public ZToolButton
{
    Q_OBJECT
public:
    ZObjectButton(const QIcon& icon, const QString& text, QWidget* parent = nullptr);
    ~ZObjectButton();

protected:
    QBrush backgrondColor(QStyle::State state) const override;
};

class ZMiniToolButton : public ZToolButton
{
    Q_OBJECT
public:
    ZMiniToolButton(const QIcon& icon, QWidget* parent = nullptr);

protected:
    QBrush backgrondColor(QStyle::State state) const override;
};

#endif