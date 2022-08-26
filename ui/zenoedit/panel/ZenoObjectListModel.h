//
// Created by zh on 2022/8/26.
//

#ifndef ZENO_ZENOOBJECTLISTMODEL_H
#define ZENO_ZENOOBJECTLISTMODEL_H

#include <QAbstractListModel>
#include "zeno/core/IObject.h"

class ZenoObjectListModel: public QAbstractListModel {
    int rowCount(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    std::vector<std::pair<std::string, zeno::IObject*>> obj_list;
public:
    void updateByObjectsMan();
};


#endif //ZENO_ZENOOBJECTLISTMODEL_H
