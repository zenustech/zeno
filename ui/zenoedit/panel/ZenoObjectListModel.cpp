//
// Created by zh on 2022/8/26.
//

#include "ZenoObjectListModel.h"
#include "viewport/zenovis.h"
#include <zenovis/ObjectsManager.h>

int ZenoObjectListModel::rowCount(const QModelIndex &parent) const {
    return obj_list.size();
}

QVariant ZenoObjectListModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid()) {
        return {};
    }
    if (role == Qt::DisplayRole) {
        return QString(obj_list.at(index.row()).first.c_str());
    }
    return {};
}

void ZenoObjectListModel::updateByObjectsMan() {
    beginResetModel();
    obj_list.clear();
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    obj_list = scene->objectsMan->pairs();
    endResetModel();
}
