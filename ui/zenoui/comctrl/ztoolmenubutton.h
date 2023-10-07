#ifndef __ZTOOLMENUBUTTON_H__
#define __ZTOOLMENUBUTTON_H__

class ZStyleOptionToolButton;

#include <QtWidgets>
#include "ztoolbutton.h"

class ZToolMenuButton : public ZToolButton {
    Q_OBJECT
public:
    ZToolMenuButton(QWidget *parent = nullptr);
    void addAction(const QString& action, const QString& icon = "");
signals:
    void textChanged();
protected:
    virtual void mouseReleaseEvent(QMouseEvent* e) override;
    virtual QSize sizeHint() const override;
private:
    QMenu* m_pMenu;
};

#endif
