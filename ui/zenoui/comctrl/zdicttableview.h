#ifndef __ZDICT_TABLEVIEW_H__
#define __ZDICT_TABLEVIEW_H__

#include <QtWidgets>

class ZDictTableView : public QTableView
{
    Q_OBJECT
public:
    ZDictTableView(QWidget* parent = nullptr);

protected:
    void keyReleaseEvent(QKeyEvent* event);
};


#endif