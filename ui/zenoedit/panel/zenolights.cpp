#include "zenolights.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/modelrole.h>
#include "viewport/zenovis.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "zeno/utils/log.h"
#include "zeno/core/Session.h"
#include <zeno/types/PrimitiveObject.h>
#include <zenoui/comctrl/zcombobox.h>
#include <zeno/extra/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <glm/glm.hpp>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalComm.h>


ZenoLights::ZenoLights(QWidget *parent) : QWidget(parent) {
    QVBoxLayout* pMainLayout = new QVBoxLayout;
    pMainLayout->setContentsMargins(QMargins(0, 0, 0, 0));
    setLayout(pMainLayout);
    setFocusPolicy(Qt::ClickFocus);

    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(37, 37, 38));
    setPalette(palette);
    setAutoFillBackground(true);

    //setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);

    QHBoxLayout* pSunLightLayout = new QHBoxLayout;
    pMainLayout->addLayout(pSunLightLayout);

    QLabel* sunLongitudeLabel = new QLabel(tr("Longitude: "));
    sunLongitudeLabel->setProperty("cssClass", "proppanel");
    pSunLightLayout->addWidget(sunLongitudeLabel);

    sunLongitude->setProperty("cssClass", "proppanel");
    sunLongitude->setNumSlider({ .1, 1, 10 });
    pSunLightLayout->addWidget(sunLongitude);

    QLabel* sunLatitudeLabel = new QLabel(tr("Latitude: "));
    sunLatitudeLabel->setProperty("cssClass", "proppanel");
    pSunLightLayout->addWidget(sunLatitudeLabel);

    sunLatitude->setProperty("cssClass", "proppanel");
    sunLatitude->setNumSlider({ .1, 1, 10 });
    pSunLightLayout->addWidget(sunLatitude);

    QLabel* sunSoftnessLabel = new QLabel(tr("Softness: "));
    sunSoftnessLabel->setProperty("cssClass", "proppanel");
    pSunLightLayout->addWidget(sunSoftnessLabel);

    sunSoftness->setProperty("cssClass", "proppanel");
    sunSoftness->setNumSlider({ 0.01, .1 });
    pSunLightLayout->addWidget(sunSoftness);

    QHBoxLayout* pWindLayout = new QHBoxLayout;
    pMainLayout->addLayout(pWindLayout);

    QLabel* sunWindLongLabel = new QLabel(tr("WindLong: "));
    sunWindLongLabel->setProperty("cssClass", "proppanel");
    pWindLayout->addWidget(sunWindLongLabel);

    windLong->setProperty("cssClass", "proppanel");
    windLong->setNumSlider({ .1, 1, 10 });
    pWindLayout->addWidget(windLong);

    QLabel* sunWindLatLabel = new QLabel(tr("WindLat: "));
    sunWindLatLabel->setProperty("cssClass", "proppanel");
    pWindLayout->addWidget(sunWindLatLabel);

    windLat->setProperty("cssClass", "proppanel");
    windLat->setNumSlider({ .1, 1, 10 });
    pWindLayout->addWidget(windLat);

    QLabel* timeStartLabel = new QLabel(tr("TimeStart: "));
    timeStartLabel->setProperty("cssClass", "proppanel");
    pWindLayout->addWidget(timeStartLabel);

    timeStart->setProperty("cssClass", "proppanel");
    timeStart->setNumSlider({ .1, 1, 10 });
    pWindLayout->addWidget(timeStart);

    QLabel* timeSpeedLabel = new QLabel(tr("TimeSpeed: "));
    timeSpeedLabel->setProperty("cssClass", "proppanel");
    pWindLayout->addWidget(timeSpeedLabel);

    timeSpeed->setProperty("cssClass", "proppanel");
    timeSpeed->setNumSlider({ .1, 1, 10 });
    pWindLayout->addWidget(timeSpeed);

    QHBoxLayout* pLightLayout = new QHBoxLayout;
    pMainLayout->addLayout(pLightLayout);

    QLabel* sunLightIntensityLabel = new QLabel(tr("LightIntensity: "));
    sunLightIntensityLabel->setProperty("cssClass", "proppanel");
    pLightLayout->addWidget(sunLightIntensityLabel);

    sunLightIntensity->setProperty("cssClass", "proppanel");
    sunLightIntensity->setNumSlider({ .1, 1, 10 });
    pLightLayout->addWidget(sunLightIntensity);

    QLabel* colorTemperatureMixLabel = new QLabel(tr("colorTemperatureMix: "));
    colorTemperatureMixLabel->setProperty("cssClass", "proppanel");
    pLightLayout->addWidget(colorTemperatureMixLabel);

    colorTemperatureMix->setProperty("cssClass", "proppanel");
    colorTemperatureMix->setNumSlider({ .1, 1, 10 });
    pLightLayout->addWidget(colorTemperatureMix);

    QLabel* colorTemperatureLabel = new QLabel(tr("colorTemperature: "));
    colorTemperatureLabel->setProperty("cssClass", "proppanel");
    pLightLayout->addWidget(colorTemperatureLabel);

    colorTemperature->setProperty("cssClass", "proppanel");
    colorTemperature->setNumSlider({ 10, 100 });
    pLightLayout->addWidget(colorTemperature);

    QHBoxLayout* pTitleLayout = new QHBoxLayout;

    write_btn->setProperty("cssClass", "grayButton");
    write_all_btn->setProperty("cssClass", "grayButton");
    procedural_sky_btn->setProperty("cssClass", "grayButton");
    sync_btn->setProperty("cssClass", "grayButton");
    connect(write_btn, &QPushButton::clicked, this, [&](){
        QModelIndex index = lights_view->currentIndex();
        if (index.row() >= 0) {
            QString primid = index.data(Qt::DisplayRole).toString();
            if (primid.contains("LightNode")) {
                write_param_into_node(primid);
            }
        }
    });
    connect(write_all_btn, &QPushButton::clicked, this, [&](){


        DisplayWidget* pWid = getViewportWithOptixFirst();
        if (!pWid)
            return;
        auto pZenovis = pWid->getZenoVis();
        ZASSERT_EXIT(pZenovis);
        auto scene = pZenovis->getSession()->get_scene();
        ZASSERT_EXIT(scene);

        //todo: move objsMan outof scene.
        std::string primid = zeno::getSession().globalComm->getLightObjKeyByLightObjID("LightNode");
        write_param_into_node(QString(primid.c_str()));
    });
    connect(procedural_sky_btn, &QPushButton::clicked, this, [&](){

        DisplayWidget* pWid = getViewportWithOptixFirst();
        if (!pWid)
            return;

        auto pZenovis = pWid->getZenoVis();
        ZASSERT_EXIT(pZenovis);
        auto scene = pZenovis->getSession()->get_scene();
        ZASSERT_EXIT(scene);

        std::string primid = zeno::getSession().globalComm->getLightObjKeyByLightObjID("ProceduralSky");
        write_param_into_node(QString(primid.c_str()));
    });
    connect(sync_btn, &QPushButton::clicked, this, [=]() {
        dataModel->updateByObjectsMan();
    });

    pTitleLayout->addWidget(write_btn);
    pTitleLayout->addWidget(write_all_btn);
    pTitleLayout->addWidget(procedural_sky_btn);
    pTitleLayout->addWidget(sync_btn);

    pPrimName->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pPrimName);

    pMainLayout->addLayout(pTitleLayout);

    lights_view->setAlternatingRowColors(true);
    lights_view->setProperty("cssClass", "proppanel");
    lights_view->setModel(this->dataModel);
    pMainLayout->addWidget(lights_view);

    zenoApp->getMainWindow()->lightPanel = this;

    connect(lights_view, &QListView::pressed, this, [&](auto & index){
        std::string name = this->dataModel->light_names[index.row()];

        DisplayWidget* pWid = getViewportWithOptixFirst();
        if (!pWid) {
            return;
        }
        auto pZenovis = pWid->getZenoVis();
        ZASSERT_EXIT(pZenovis);
        auto scene = pZenovis->getSession()->get_scene();
        ZASSERT_EXIT(scene);

        zeno::vec3f pos;
        zeno::vec3f scale;
        zeno::vec3f rotate;
        zeno::vec3f clr;
        float intensity;
        zeno::getSession().globalComm->getLightObjData(name, pos, scale, rotate, clr, intensity);

        posXEdit->setText(QString::number(pos[0]));
        posYEdit->setText(QString::number(pos[1]));
        posZEdit->setText(QString::number(pos[2]));

        scaleXEdit->setText(QString::number(scale[0]));
        scaleYEdit->setText(QString::number(scale[1]));
        scaleZEdit->setText(QString::number(scale[2]));

        rotateXEdit->setText(QString::number(rotate[0]));
        rotateYEdit->setText(QString::number(rotate[1]));
        rotateZEdit->setText(QString::number(rotate[2]));

        colorXEdit->setText(QString::number(clr[0]));
        colorYEdit->setText(QString::number(clr[1]));
        colorZEdit->setText(QString::number(clr[2]));

        intensityEdit->setText(QString::number(intensity));
    });

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

        posXEdit->setNumSlider({ .0001, .001, .01, .1, 1, 10, 100 });
        posXEdit->setProperty("cssClass", "proppanel");
        posXEdit->setValidator(new QDoubleValidator);
        posYEdit->setNumSlider({ .0001, .001, .01, .1, 1, 10, 100 });
        posYEdit->setProperty("cssClass", "proppanel");
        posYEdit->setValidator(new QDoubleValidator);
        posZEdit->setNumSlider({ .0001, .001, .01, .1, 1, 10, 100 });
        posZEdit->setProperty("cssClass", "proppanel");
        posZEdit->setValidator(new QDoubleValidator);
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

        scaleXEdit->setNumSlider({ .0001, .001, .01, .1, 1, 10, 100 });
        scaleXEdit->setProperty("cssClass", "proppanel");
        scaleXEdit->setValidator(new QDoubleValidator);
        scaleYEdit->setNumSlider({ .0001, .001, .01, .1, 1, 10, 100 });
        scaleYEdit->setProperty("cssClass", "proppanel");
        scaleYEdit->setValidator(new QDoubleValidator);
        scaleZEdit->setNumSlider({ .0001, .001, .01, .1, 1, 10, 100 });
        scaleZEdit->setProperty("cssClass", "proppanel");
        scaleZEdit->setValidator(new QDoubleValidator);
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

        rotateXEdit->setNumSlider({ .1, 1, 10, });
        rotateXEdit->setProperty("cssClass", "proppanel");
        rotateXEdit->setValidator(new QDoubleValidator);
        rotateYEdit->setNumSlider({ .1, 1, 10, });
        rotateYEdit->setProperty("cssClass", "proppanel");
        rotateYEdit->setValidator(new QDoubleValidator);
        rotateZEdit->setNumSlider({ .1, 1, 10, });
        rotateZEdit->setProperty("cssClass", "proppanel");
        rotateZEdit->setValidator(new QDoubleValidator);
    }

    {
        QHBoxLayout* pColorLayout = new QHBoxLayout();
        QLabel* colorHeader = new QLabel("Color: ");
        colorHeader->setProperty("cssClass", "proppanel");
        pColorLayout->addWidget(colorHeader);
        QLabel* colorX = new QLabel(" x: ");
        colorX->setProperty("cssClass", "proppanel");
        pColorLayout->addWidget(colorX);
        colorXEdit->setProperty("cssClass", "proppanel");
        pColorLayout->addWidget(colorXEdit);
        QLabel* colorY = new QLabel(" y: ");
        colorY->setProperty("cssClass", "proppanel");
        pColorLayout->addWidget(colorY);
        colorYEdit->setProperty("cssClass", "proppanel");
        pColorLayout->addWidget(colorYEdit);
        QLabel* colorZ = new QLabel(" z: ");
        colorZ->setProperty("cssClass", "proppanel");
        pColorLayout->addWidget(colorZ);
        colorZEdit->setProperty("cssClass", "proppanel");
        pColorLayout->addWidget(colorZEdit);
        pMainLayout->addLayout(pColorLayout);

        colorXEdit->setNumSlider({ .01, .1,});
        colorXEdit->setProperty("cssClass", "proppanel");
        colorXEdit->setValidator(new QDoubleValidator);
        colorYEdit->setNumSlider({ .01, .1, });
        colorYEdit->setProperty("cssClass", "proppanel");
        colorYEdit->setValidator(new QDoubleValidator);
        colorZEdit->setNumSlider({ .01, .1, });
        colorZEdit->setProperty("cssClass", "proppanel");
        colorZEdit->setValidator(new QDoubleValidator);
    }

    {
        QHBoxLayout* pMouseSenLayout = new QHBoxLayout();
        QLabel* mouseSen = new QLabel("Intensity: ");
        mouseSen->setProperty("cssClass", "proppanel");
        pMouseSenLayout->addWidget(mouseSen);
        QLabel* mouseSenValue = new QLabel(" v: ");
        mouseSenValue->setProperty("cssClass", "proppanel");
        pMouseSenLayout->addWidget(mouseSenValue);
        intensityEdit->setProperty("cssClass", "proppanel");
        pMouseSenLayout->addWidget(intensityEdit);

        pMainLayout->addLayout(pMouseSenLayout);

        intensityEdit->setNumSlider({  1, 10, 100, });
        intensityEdit->setProperty("cssClass", "proppanel");
        intensityEdit->setValidator(new QDoubleValidator);
    }

    {
        QHBoxLayout* pCamAperture = new QHBoxLayout();
        QLabel* camAperture = new QLabel("CameraAperture: ");
        camAperture->setProperty("cssClass", "proppanel");
        pCamAperture->addWidget(camAperture);
        QLabel* cav = new QLabel(" v: ");
        cav->setProperty("cssClass", "proppanel");
        pCamAperture->addWidget(cav);
        camApertureEdit->setProperty("cssClass", "proppanel");
        pCamAperture->addWidget(camApertureEdit);

        pMainLayout->addLayout(pCamAperture);

        camApertureEdit->setNumSlider({ .0001, .001, .01, .1, 1, 10, 100 });
        camApertureEdit->setProperty("cssClass", "proppanel");
        camApertureEdit->setValidator(new QDoubleValidator);
    }

    {
        QHBoxLayout* pCamDisPlane = new QHBoxLayout();
        QLabel* camDisPlane = new QLabel("CameraDistancePlane: ");
        camDisPlane->setProperty("cssClass", "proppanel");
        pCamDisPlane->addWidget(camDisPlane);
        QLabel* cdpv = new QLabel(" v: ");
        cdpv->setProperty("cssClass", "proppanel");
        pCamDisPlane->addWidget(cdpv);
        camDisPlaneEdit->setProperty("cssClass", "proppanel");
        pCamDisPlane->addWidget(camDisPlaneEdit);

        pMainLayout->addLayout(pCamDisPlane);

        camDisPlaneEdit->setNumSlider({ .0001, .001, .01, .1, 1, 10, 100 });
        camDisPlaneEdit->setProperty("cssClass", "proppanel");
        camDisPlaneEdit->setValidator(new QDoubleValidator);
    }

    pStatusBar->setProperty("cssClass", "proppanel");
    pMainLayout->addWidget(pStatusBar);

    connect(sunLatitude, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });
    connect(sunLongitude, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });
    connect(sunSoftness, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });
    connect(timeStart, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });
    connect(timeSpeed, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });
    connect(windLong, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });
    connect(windLat, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });
    connect(sunLightIntensity, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });
    connect(colorTemperatureMix, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });
    connect(colorTemperature, &QLineEdit::textChanged, this, [&](){ modifySunLightDir(); });

    connect(posXEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });
    connect(posYEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });
    connect(posZEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });

    connect(rotateXEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });
    connect(rotateYEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });
    connect(rotateZEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });

    connect(scaleXEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });
    connect(scaleYEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });
    connect(scaleZEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });

    connect(colorXEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });
    connect(colorYEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });
    connect(colorZEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });

    connect(intensityEdit, &QLineEdit::textChanged, this, [&](){ modifyLightData(); });

    connect(camApertureEdit, &QLineEdit::textChanged, this, [&](){
        QVector<DisplayWidget*> views = zenoApp->getMainWindow()->viewports();
        for (auto pDisplay : views) {
            ZASSERT_EXIT(pDisplay);
            pDisplay->updateCameraProp(camApertureEdit->text().toFloat(), camDisPlaneEdit->text().toFloat());
            zenoApp->getMainWindow()->updateViewport();
        }
    });
    connect(camDisPlaneEdit, &QLineEdit::textChanged, this, [&](){
        QVector<DisplayWidget*> views = zenoApp->getMainWindow()->viewports();
        for (auto pDisplay : views) {
            ZASSERT_EXIT(pDisplay);
            pDisplay->updateCameraProp(camApertureEdit->text().toFloat(), camDisPlaneEdit->text().toFloat());
            zenoApp->getMainWindow()->updateViewport();
        }
    });

    updateLights();
}

DisplayWidget* ZenoLights::getViewportWithOptixFirst() const
{
    ZenoMainWindow* pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin, nullptr);
    DisplayWidget *pWid = pWin->getOptixWidget();
    if (!pWid) {
        QVector<DisplayWidget*> views = pWin->viewports();
        if (!views.isEmpty()) {
            pWid = views[0];
        }
    }
    return pWid;
}

void ZenoLights::updateLights() {
    dataModel->updateByObjectsMan();
}

std::vector<zeno::vec3f> ZenoLights::computeLightPrim(zeno::vec3f position, zeno::vec3f rotate, zeno::vec3f scale){
    auto start_point = zeno::vec3f(0.5, 0, 0.5);
    float rm = 1.0f;
    float cm = 1.0f;
    std::vector<zeno::vec3f> verts;
    float ax = rotate[0] * (3.14159265358979323846 / 180.0);
    float ay = rotate[1] * (3.14159265358979323846 / 180.0);
    float az = rotate[2] * (3.14159265358979323846 / 180.0);
    glm::mat3 mx = glm::mat3(1, 0, 0, 0, cos(ax), -sin(ax), 0, sin(ax), cos(ax));
    glm::mat3 my = glm::mat3(cos(ay), 0, sin(ay), 0, 1, 0, -sin(ay), 0, cos(ay));
    glm::mat3 mz = glm::mat3(cos(az), -sin(az), 0, sin(az), cos(az), 0, 0, 0, 1);

    for(int i=0; i<=1; i++){
        auto rp = start_point - zeno::vec3f(i*rm, 0, 0);
        for(int j=0; j<=1; j++){
            auto p = rp - zeno::vec3f(0, 0, j*cm);
            // S R T
            p = p * scale;
            auto gp = glm::vec3(p[0], p[1], p[2]);
            gp = mz * my * mx * gp;
            p = zeno::vec3f(gp.x, gp.y, gp.z);
            auto zcp = zeno::vec3f(p[0], p[1], p[2]);
            zcp = zcp + position;

            verts.push_back(zcp);
        }
    }

    return verts;
}

void ZenoLights::modifyLightData() {
    auto index = this->lights_view->currentIndex();
    //printf("modifyLightData %d\n", index.row());
    if (index.row() == -1) {
        return;
    }
    float posX = posXEdit->text().toFloat();
    float posY = posYEdit->text().toFloat();
    float posZ = posZEdit->text().toFloat();
    float scaleX = scaleXEdit->text().toFloat();
    float scaleY = scaleYEdit->text().toFloat();
    float scaleZ = scaleZEdit->text().toFloat();
    float rotateX = rotateXEdit->text().toFloat();
    float rotateY = rotateYEdit->text().toFloat();
    float rotateZ = rotateZEdit->text().toFloat();

    float r = colorXEdit->text().toFloat();
    float g = colorYEdit->text().toFloat();
    float b = colorZEdit->text().toFloat();

    float intensity = this->intensityEdit->text().toFloat();
    std::string name = this->dataModel->light_names[index.row()];

    zeno::vec3f pos = zeno::vec3f(posX, posY, posZ);
    zeno::vec3f scale = zeno::vec3f(scaleX, scaleY, scaleZ);
    zeno::vec3f rotate = zeno::vec3f(rotateX, rotateY, rotateZ);
    auto verts = computeLightPrim(pos, rotate, scale);

    ZenoMainWindow* pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);

    QVector<DisplayWidget*> views = pWin->viewports();
    for (auto pDisplay : views)
    {
        if (pDisplay->isGLViewport())
            continue;

        bool ret = zeno::getSession().globalComm->setLightObjData(name, pos, scale, rotate, zeno::vec3f(r, g, b), intensity, verts);
        if (ret)
        {
            pDisplay->setSimpleRenderOption();
            zenoApp->getMainWindow()->updateViewport();
        }
    }
}

void ZenoLights::modifySunLightDir() {
    float sunLongitudeValue = sunLongitude->text().toFloat();
    float sunLatitudeValue = sunLatitude->text().toFloat();
    zeno::vec2f sunLightDir = zeno::vec2f(sunLongitudeValue, sunLatitudeValue);

    float windLongValue = windLong->text().toFloat();
    float windLatValue = windLat->text().toFloat();
    zeno::vec2f windDir = zeno::vec2f(windLongValue, windLatValue);

    float sunSoftnessValue = sunSoftness->text().toFloat();
    float timeStartValue = timeStart->text().toFloat();
    float timeSpeedValue = timeSpeed->text().toFloat();

    float sunLightIntensityValue = sunLightIntensity->text().toFloat();
    float colorTemperatureMixValue = colorTemperatureMix->text().toFloat();
    float colorTemperatureValue = colorTemperature->text().toFloat();

    ZenoMainWindow* pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);

    QVector<DisplayWidget*> views = pWin->viewports();
    for (auto pDisplayWid : views)
    {
        zeno::getSession().globalComm->setProceduralSkyData("", sunLightDir, sunSoftnessValue, windDir, timeStartValue, timeSpeedValue, sunLightIntensityValue, colorTemperatureMixValue, colorTemperatureValue);
        pDisplayWid->setSimpleRenderOption();
        zenoApp->getMainWindow()->updateViewport();
    }
}

void ZenoLights::write_param_into_node(const QString& primid) {

    ZenoMainWindow *pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);
    QVector<DisplayWidget*> views = pWin->viewports();
    for (DisplayWidget* pDisplay : views)
    {
        auto pZenovis = pDisplay->getZenoVis();
        ZASSERT_EXIT(pZenovis);
        auto scene = pZenovis->getSession()->get_scene();

        IGraphsModel* pIGraphsModel = zenoApp->graphsManagment()->currentModel();
        if (pIGraphsModel == nullptr) {
            return;
        }
        auto subgraphIndices = pIGraphsModel->subgraphsIndice();

        for (const auto& subGpIdx : subgraphIndices) {
            int n = pIGraphsModel->itemCount(subGpIdx);
            for (int i = 0; i < n; i++) {
                const NODE_DATA& item = pIGraphsModel->itemData(pIGraphsModel->index(i, subGpIdx), subGpIdx);
                if (item[ROLE_OBJID].toString().contains(primid.split(':').front())) {
                    auto inputs = item[ROLE_INPUTS].value<INPUT_SOCKETS>();
                    zeno::vec3f p;
                    zeno::vec3f s;
                    zeno::vec3f r;
                    zeno::vec3f c;
                    float intensity;
                    bool ret = zeno::getSession().globalComm->getLightObjData(primid.toStdString(), p, s, r, c, intensity);
                    if (ret)
                    {
                        inputs["position"].info.defaultValue.setValue(UI_VECTYPE({ p[0], p[1], p[2] }));
                        inputs["scale"].info.defaultValue.setValue(UI_VECTYPE({ s[0], s[1], s[2] }));
                        inputs["rotate"].info.defaultValue.setValue(UI_VECTYPE({ r[0], r[1], r[2] }));
                        inputs["color"].info.defaultValue.setValue(UI_VECTYPE({ c[0], c[1], c[2] }));
                        inputs["intensity"].info.defaultValue = (double)intensity;
                        auto nodeIndex = pIGraphsModel->index(item[ROLE_OBJID].toString(), subGpIdx);
                        pIGraphsModel->setNodeData(nodeIndex, subGpIdx, QVariant::fromValue(inputs), ROLE_INPUTS);
                        continue;
                    }
                    zeno::vec2f d;
                    zeno::vec2f w;
                    float sunLightSoftness;
                    float timeStart;
                    float timeSpeed;
                    float sunLightIntensity;
                    float colorTemperatureMix;
                    float colorTemperature;
                    ret = zeno::getSession().globalComm->getProceduralSkyData(primid.toStdString(), d, sunLightSoftness, w, timeStart, timeSpeed, sunLightIntensity, colorTemperatureMix, colorTemperature);
                    if(ret) {
                        inputs["sunLightDir"].info.defaultValue.setValue(UI_VECTYPE({ d[0], d[1] }));
                        inputs["windDir"].info.defaultValue.setValue(UI_VECTYPE({ w[0], w[1] }));
                        inputs["sunLightSoftness"].info.defaultValue = (double)sunLightSoftness;
                        inputs["timeStart"].info.defaultValue = (double)timeStart;
                        inputs["timeSpeed"].info.defaultValue = (double)timeSpeed;
                        inputs["sunLightIntensity"].info.defaultValue = (double)sunLightIntensity;
                        inputs["colorTemperatureMix"].info.defaultValue = (double)colorTemperatureMix;
                        inputs["colorTemperature"].info.defaultValue = (double)colorTemperature;
                        auto nodeIndex = pIGraphsModel->index(item[ROLE_OBJID].toString(), subGpIdx);
                        pIGraphsModel->setNodeData(nodeIndex, subGpIdx, QVariant::fromValue(inputs), ROLE_INPUTS);
                    }
                }
            }
        }
    }
}
