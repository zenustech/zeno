//
// Created by zh on 2022/6/27.
//

#ifndef ZENO_ZENOLIGHTS_H
#define ZENO_ZENOLIGHTS_H

#include <QtWidgets>
#include "PrimAttrTableModel.h"
#include "ZLightsModel.h"
#include <zenoui/comctrl/zlineedit.h>

class DisplayWidget;

class ZenoLights : public QWidget {
    Q_OBJECT
public:
    ZLineEdit* sunLongitude = new ZLineEdit("-60");
    ZLineEdit* sunLatitude = new ZLineEdit("45");
    ZLineEdit* sunSoftness = new ZLineEdit("1");
    ZLineEdit* windLong = new ZLineEdit("0");
    ZLineEdit* windLat = new ZLineEdit("0");
    ZLineEdit* timeStart = new ZLineEdit("0");
    ZLineEdit* timeSpeed = new ZLineEdit("0.1");
    ZLineEdit* sunLightIntensity = new ZLineEdit("1");
    ZLineEdit* colorTemperatureMix = new ZLineEdit("0");
    ZLineEdit* colorTemperature = new ZLineEdit("6500");
    QPushButton* write_btn = new QPushButton("Write");
    QPushButton* write_all_btn = new QPushButton("Write ALL");
    QPushButton* procedural_sky_btn = new QPushButton("Procedural Sky");
    QPushButton* sync_btn = new QPushButton("Sync Lights");

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

    ZLineEdit* camApertureEdit = new ZLineEdit("0.0");
    ZLineEdit* camDisPlaneEdit = new ZLineEdit("2.0");
    ZLineEdit* intensityEdit = new ZLineEdit("1");
    void modifyLightData();
    void modifySunLightDir();
    void write_param_into_node(const QString& primid);

public:
    ZenoLights(QWidget* parent = nullptr);
    void updateLights();
    static std::vector<zeno::vec3f> computeLightPrim(zeno::vec3f position, zeno::vec3f rotate, zeno::vec3f scale);

private:
    DisplayWidget* getViewportWithOptixFirst() const;
};



#endif //ZENO_ZENOLIGHTS_H
