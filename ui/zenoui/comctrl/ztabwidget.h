#ifndef __ZTABWIDGET_H__
#define __ZTABWIDGET_H__

#include <QtWidgets>
#include "ztabbar.h"

class ZTabWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ZTabWidget(QWidget* parent = nullptr);
    ~ZTabWidget();

private:
    void init();

    QStackedWidget* m_stack;
    ZTabBar* m_pTabbar;
};


#endif