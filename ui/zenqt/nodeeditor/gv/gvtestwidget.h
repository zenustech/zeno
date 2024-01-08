#ifndef __GVTEST_WIDGET_H__
#define __GVTEST_WIDGET_H__

#include <QtWidgets>

class TestGraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
    TestGraphicsView(QWidget* parent = nullptr);
};


#endif