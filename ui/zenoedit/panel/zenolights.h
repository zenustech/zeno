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

    QLabel* pStatusBar = new QLabel();
    QLabel* pPrimName = new QLabel();
    QListView *lights_view = new QListView();
    ZLightsModel* dataModel = new ZLightsModel();

    QLineEdit* posXEdit = new QLineEdit();
    QLineEdit* posYEdit = new QLineEdit();
    QLineEdit* posZEdit = new QLineEdit();

    QLineEdit* scaleXEdit = new QLineEdit();
    QLineEdit* scaleYEdit = new QLineEdit();
    QLineEdit* scaleZEdit = new QLineEdit();

    QLineEdit* rotateXEdit = new QLineEdit();
    QLineEdit* rotateYEdit = new QLineEdit();
    QLineEdit* rotateZEdit = new QLineEdit();

    QLineEdit* colorXEdit = new QLineEdit();
    QLineEdit* colorYEdit = new QLineEdit();
    QLineEdit* colorZEdit = new QLineEdit();

    void modifyLightData();

public:
    ZenoLights(QWidget* parent = nullptr);
    void updateLights();
    std::vector<zeno::vec3f> ZenoLights::computeLightPrim(zeno::vec3f position, zeno::vec3f rotate, zeno::vec3f scale);
};



#endif //ZENO_ZENOLIGHTS_H
