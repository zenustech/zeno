//
// Created by zhouhang on 2022/8/22.
//

#include "ZLightsModel.h"
#include "viewport/zenovis.h"
#include "zeno/utils/log.h"
#include <zeno/types/UserData.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/StageManager.h>
#include <zeno/types/PrimitiveObject.h>

int ZLightsModel::rowCount(const QModelIndex &parent) const {
    return light_names.size();
}

QVariant ZLightsModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid()) {
        return {};
    }
    if (role == Qt::DisplayRole) {
        return QString(light_names.at(index.row()).c_str());
    }
    return {};
}

void ZLightsModel::updateByObjectsMan() {
    beginResetModel();
    light_names.clear();
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    // USD
    //for (auto const &[key, ptr]: scene->objectsMan->lightObjects) {
    for (auto const &[key, ptr]: scene->stageMan->zenoLightObjects) {
        if (ptr->userData().get2<int>("isRealTimeObject", 0)) {
            light_names.push_back(key);
        }
    }
    endResetModel();
}
