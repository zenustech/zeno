#ifndef __ZTABBAR_H__
#define __ZTABBAR_H__

#include <QtWidgets>

struct Tab {
    QIcon icon;
    QString text;
    QRect rect;
    QToolButton* pCloseButton;
};

class ZTabBar : public QWidget
{
    Q_OBJECT
public:
    explicit ZTabBar(QWidget* parent = nullptr);
    ~ZTabBar();

    Tab* at(int index);
    void moveTab(int srcIndex, int dstIndex);

    QList<Tab> tabList;
};



#endif