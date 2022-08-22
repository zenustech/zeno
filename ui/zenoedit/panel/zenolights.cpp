#include "zenolights.h"
#include <zeno/types/PrimitiveObject.h>
#include <zenoui/comctrl/zcombobox.h>

ZenoLights::ZenoLights(QWidget *parent) : QWidget(parent) {
    QVBoxLayout* pMainLayout = new QVBoxLayout;
    pMainLayout->setContentsMargins(QMargins(0, 0, 0, 0));
    setLayout(pMainLayout);
    setFocusPolicy(Qt::ClickFocus);

    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(37, 37, 38));
    setPalette(palette);
    setAutoFillBackground(true);

    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);

    QHBoxLayout* pTitleLayout = new QHBoxLayout;

    QLabel* pPrim = new QLabel(tr("Light: "));
    pPrim->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pPrim);

    pPrimName->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pPrimName);
    pTitleLayout->addWidget(pAddLight);
    pTitleLayout->addWidget(pRemoveLight);

    pMainLayout->addLayout(pTitleLayout);

    lights_view->setAlternatingRowColors(true);
    lights_view->setProperty("cssClass", "proppanel");
    lights_view->setModel(this->dataModel);
    pMainLayout->addWidget(lights_view);

    {
        QHBoxLayout* pPosLayout = new QHBoxLayout();
        QLabel* posHeader = new QLabel("Position: ");
        posHeader->setProperty("cssClass", "proppanel");
        pPosLayout->addWidget(posHeader);
        QLabel* posX = new QLabel(" x: ");
        posX->setProperty("cssClass", "proppanel");
        pPosLayout->addWidget(posX);
        posXEdit->setProperty("cssClass", "proppanel");
        pPosLayout->addWidget(posXEdit);
        QLabel* posY = new QLabel(" y: ");
        posY->setProperty("cssClass", "proppanel");
        pPosLayout->addWidget(posY);
        posYEdit->setProperty("cssClass", "proppanel");
        pPosLayout->addWidget(posYEdit);
        QLabel* posZ = new QLabel(" z: ");
        posZ->setProperty("cssClass", "proppanel");
        pPosLayout->addWidget(posZ);
        posZEdit->setProperty("cssClass", "proppanel");
        pPosLayout->addWidget(posZEdit);
        pMainLayout->addLayout(pPosLayout);
    }

    {
        QHBoxLayout* pScaleLayout = new QHBoxLayout();
        QLabel* scaleHeader = new QLabel("ScaleSize: ");
        scaleHeader->setProperty("cssClass", "proppanel");
        pScaleLayout->addWidget(scaleHeader);
        QLabel* scaleX = new QLabel(" x: ");
        scaleX->setProperty("cssClass", "proppanel");
        pScaleLayout->addWidget(scaleX);
        scaleXEdit->setProperty("cssClass", "proppanel");
        pScaleLayout->addWidget(scaleXEdit);
        QLabel* scaleY = new QLabel(" y: ");
        scaleY->setProperty("cssClass", "proppanel");
        pScaleLayout->addWidget(scaleY);
        scaleYEdit->setProperty("cssClass", "proppanel");
        pScaleLayout->addWidget(scaleYEdit);
        QLabel* scaleZ = new QLabel(" z: ");
        scaleZ->setProperty("cssClass", "proppanel");
        pScaleLayout->addWidget(scaleZ);
        scaleZEdit->setProperty("cssClass", "proppanel");
        pScaleLayout->addWidget(scaleZEdit);
        pMainLayout->addLayout(pScaleLayout);
    }

    {
        QHBoxLayout* pRotateLayout = new QHBoxLayout();
        QLabel* rotateHeader = new QLabel("Rotate: ");
        rotateHeader->setProperty("cssClass", "proppanel");
        pRotateLayout->addWidget(rotateHeader);
        QLabel* rotateX = new QLabel(" x: ");
        rotateX->setProperty("cssClass", "proppanel");
        pRotateLayout->addWidget(rotateX);
        rotateXEdit->setProperty("cssClass", "proppanel");
        pRotateLayout->addWidget(rotateXEdit);
        QLabel* rotateY = new QLabel(" y: ");
        rotateY->setProperty("cssClass", "proppanel");
        pRotateLayout->addWidget(rotateY);
        rotateYEdit->setProperty("cssClass", "proppanel");
        pRotateLayout->addWidget(rotateYEdit);
        QLabel* rotateZ = new QLabel(" z: ");
        rotateZ->setProperty("cssClass", "proppanel");
        pRotateLayout->addWidget(rotateZ);
        rotateZEdit->setProperty("cssClass", "proppanel");
        pRotateLayout->addWidget(rotateZEdit);
        pMainLayout->addLayout(pRotateLayout);
    }

    pStatusBar->setProperty("cssClass", "proppanel");
    pMainLayout->addWidget(pStatusBar);
}

