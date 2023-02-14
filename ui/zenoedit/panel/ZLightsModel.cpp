//
// Created by zhouhang on 2022/8/22.
//

#include "ZLightsModel.h"
#include "viewport/zenovis.h"
#include "zeno/utils/log.h"
#include <zeno/types/UserData.h>
#include <zenovis/ObjectsManager.h>
#include <zeno/types/PrimitiveObject.h>
#include "zassert.h"


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
    auto session = Zenovis::GetInstance().getSession();
    ZERROR_EXIT(session);

    auto scene = session->get_scene();
    ZERROR_EXIT(scene);

    for (auto const &[key, ptr]: scene->objectsMan->lightObjects) {
        if (ptr->userData().get2<int>("isL", 0)) {
            light_names.push_back(key);
        }
    }
    endResetModel();
}
