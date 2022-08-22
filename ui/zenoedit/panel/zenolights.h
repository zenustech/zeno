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
    QPushButton* pAddLight = new QPushButton("Add");
    QPushButton* pRemoveLight = new QPushButton("Remove");
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

public:
    ZenoLights(QWidget* parent = nullptr);
};



#endif //ZENO_ZENOLIGHTS_H
