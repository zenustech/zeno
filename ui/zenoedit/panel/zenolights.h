//
// Created by zh on 2022/6/27.
//

#ifndef ZENO_ZENOLIGHTS_H
#define ZENO_ZENOLIGHTS_H

#include <QtWidgets>
#include "PrimAttrTableModel.h"
#include "ZLightsModel.h"
#include "comctrl/zlineedit.h"

class ZenoLights : public QWidget {
    Q_OBJECT
public:
    ZLineEdit* sunLongitude = new ZLineEdit("0");
    ZLineEdit* sunLatitude = new ZLineEdit("30");


    QPushButton* write_btn = new QPushButton("Write");
    QPushButton* write_all_btn = new QPushButton("Write ALL");

    QLabel* pStatusBar = new QLabel();
    QLabel* pPrimName = new QLabel();
    QListView *lights_view = new QListView();
    ZLightsModel* dataModel = new ZLightsModel();

    ZLineEdit* posXEdit = new ZLineEdit("0");
    ZLineEdit* posYEdit = new ZLineEdit("0");
    ZLineEdit* posZEdit = new ZLineEdit("0");

    ZLineEdit* scaleXEdit = new ZLineEdit("1");
    ZLineEdit* scaleYEdit = new ZLineEdit("1");
    ZLineEdit* scaleZEdit = new ZLineEdit("1");

    ZLineEdit* rotateXEdit = new ZLineEdit("0");
    ZLineEdit* rotateYEdit = new ZLineEdit("0");
    ZLineEdit* rotateZEdit = new ZLineEdit("0");

    ZLineEdit* colorXEdit = new ZLineEdit("1");
    ZLineEdit* colorYEdit = new ZLineEdit("1");
    ZLineEdit* colorZEdit = new ZLineEdit("1");

    ZLineEdit* camApertureEdit = new ZLineEdit("0.2");
    ZLineEdit* camDisPlaneEdit = new ZLineEdit("2.0");
    ZLineEdit* intensityEdit = new ZLineEdit("1");
    void modifyLightData();
    void modifySunLightDir();
    void write_param_into_node(const QString& primid);

public:
    ZenoLights(QWidget* parent = nullptr);
    void updateLights();
    std::vector<zeno::vec3f> computeLightPrim(zeno::vec3f position, zeno::vec3f rotate, zeno::vec3f scale);
};



#endif //ZENO_ZENOLIGHTS_H
