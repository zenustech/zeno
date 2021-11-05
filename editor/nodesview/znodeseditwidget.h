#ifndef __ZNODES_EDITWIDGET_H__
#define __ZNODES_EDITWIDGET_H__

#include <QtWidgets>

class ZNodesEditWidget : public QWidget
{
    Q_OBJECT
public:
    ZNodesEditWidget(QWidget* parent = nullptr);

private:
    void initMenu(QMenuBar* pMenu);
};


#endif