//
// Created by zh on 2024/2/22.
//

#ifndef ZENO_ZOPTIXCAMERASETTING_H
#define ZENO_ZOPTIXCAMERASETTING_H

#include "zenovis/Camera.h"
#include <QtWidgets>

class ZOptixCameraSetting : public QDialog {
public:
    ZOptixCameraSetting(zenovis::ZOptixCameraSettingInfo &info, QWidget* parent = nullptr);
};



#endif //ZENO_ZOPTIXCAMERASETTING_H
