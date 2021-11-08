#ifndef __ZMENUBUTTON_H__
#define __ZMENUBUTTON_H__

#include "ztoolbutton.h"
#include "../style/zstyleoption.h"
#include "zpopupwidget.h"

class ZMenuButton : public ZToolButton
{
    Q_OBJECT
public:
    ZMenuButton(ButtonOption option, const QIcon& icon = QIcon(), const QSize& iconSize = QSize(), const QString& text = QString(), QWidget* parent = nullptr);
    ~ZMenuButton();

signals:
    void trigger();
    void popup();
    void popout();

protected slots:
    virtual void popupChildWidget();

protected:
    virtual bool event(QEvent* e);
    void initStyleOption(ZStyleOptionToolButton* option) const;
    void paintEvent(QPaintEvent* event) override;
};

#endif
