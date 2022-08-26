//
// Created by zh on 2022/8/26.
//

#ifndef ZENO_ZENOOBJECTLIST_H
#define ZENO_ZENOOBJECTLIST_H

#include <QtWidgets>
#include "ZenoObjectListModel.h"

class ZenoObjectList : public QWidget {
    Q_OBJECT
public:
    QLabel* pStatusBar = new QLabel();
    QListView *lights_view = new QListView();
    ZenoObjectListModel* dataModel = new ZenoObjectListModel();
    ZenoObjectList(QWidget* parent = nullptr);
};


#endif //ZENO_ZENOOBJECTLIST_H
