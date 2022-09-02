//
// Created by zh on 2022/6/27.
//

#ifndef ZENO_ZENOLIGHTS_H
#define ZENO_ZENOLIGHTS_H

#include <QtWidgets>
#include "PrimAttrTableModel.h"
#include "ZLightsModel.h"

class ZenoLights : public QWidget {
    Q_OBJECT
public:
    QLabel* pStatusBar = new QLabel();
    QLabel* pPrimName = new QLabel();
    QListView *lights_view = new QListView();
    ZLightsModel* dataModel = new ZLightsModel();

    QLineEdit* posXEdit = new QLineEdit("0");
    QLineEdit* posYEdit = new QLineEdit("0");
    QLineEdit* posZEdit = new QLineEdit("0");

    QLineEdit* scaleXEdit = new QLineEdit("1");
    QLineEdit* scaleYEdit = new QLineEdit("1");
    QLineEdit* scaleZEdit = new QLineEdit("1");

    QLineEdit* rotateXEdit = new QLineEdit("0");
    QLineEdit* rotateYEdit = new QLineEdit("0");
    QLineEdit* rotateZEdit = new QLineEdit("0");

    QLineEdit* colorXEdit = new QLineEdit("1");
    QLineEdit* colorYEdit = new QLineEdit("1");
    QLineEdit* colorZEdit = new QLineEdit("1");

    QLineEdit* mouseSenEdit = new QLineEdit("0.2");
    QLineEdit* camApertureEdit = new QLineEdit("0.2");
    QLineEdit* camDisPlaneEdit = new QLineEdit("2.0");
    void modifyLightData();

public:
    ZenoLights(QWidget* parent = nullptr);
    void updateLights();
    std::vector<zeno::vec3f> computeLightPrim(zeno::vec3f position, zeno::vec3f rotate, zeno::vec3f scale);
};



#endif //ZENO_ZENOLIGHTS_H
