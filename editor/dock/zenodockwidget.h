#ifndef __ZENO_DOCKWIDGET_H__
#define __ZENO_DOCKWIDGET_H__

#include <QtWidgets>

class ZenoDockWidget : public QDockWidget
{
    Q_OBJECT
public:
    explicit ZenoDockWidget(const QString &title, QWidget *parent = nullptr,
                         Qt::WindowFlags flags = Qt::WindowFlags());
    explicit ZenoDockWidget(QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());
    ~ZenoDockWidget();
};



#endif