//
// Created by zh on 2023/3/23.
//

#include "zenoimagepanel.h"
#include "PrimAttrTableModel.h"
#include "viewport/zenovis.h"
#include "zenovis/ObjectsManager.h"
#include "zeno/utils/format.h"
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zenoui/comctrl/zcombobox.h>
#include "zeno/utils/log.h"
#include "zenoapplication.h"
#include "zassert.h"
#include "viewport/viewportwidget.h"
#include "zenomainwindow.h"
#include "viewport/displaywidget.h"


void ZenoImagePanel::clear() {
    if (image_view) {
        image_view->clearImage();
    }
    pPrimName->clear();
    pStatusBar->clear();
}

void ZenoImagePanel::setPrim(std::string primid) {
    primid = primid.substr(0, primid.find(":"));
    pPrimName->setText(primid.c_str());
    zenovis::Scene* scene = nullptr;
    auto mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    QVector<DisplayWidget*> wids = mainWin->viewports();
    if (!wids.isEmpty())
    {
        auto session = wids[0]->getZenoVis()->getSession();
        ZASSERT_EXIT(session);
        scene = session->get_scene();
    }
    if (!scene)
        return;

    bool enableGamma = pGamma->checkState() == Qt::Checked;
    bool enableAlpha = pAlpha->checkState() == Qt::Checked;
    bool found = false;
    for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
        if ((key.substr(0, key.find(":"))) != primid) {
            continue;
        }
        auto &ud = ptr->userData();
        if (ud.get2<int>("isImage", 0) == 0) {
            continue;
        }
        found = true;
        if (auto obj = dynamic_cast<zeno::PrimitiveObject *>(ptr)) {
            int width = ud.get2<int>("w");
            int height = ud.get2<int>("h");
            if (image_view) {
                QImage img(width, height, QImage::Format_RGB32);
                int gridSize = 50;
                if (obj->verts.has_attr("alpha")&&enableAlpha) {
                    auto &alpha = obj->verts.attr<float>("alpha");
                    for (auto i = 0; i < obj->verts.size(); i++) {
                        int h = i / width;
                        //int h = i % height;  check image vert order
                        int w = i % width;
                        //int w = i / height;
                        auto foreground = obj->verts[i];
                        if (enableGamma) {
                            foreground = zeno::pow(foreground, 1.0f / 2.2f);
                        }
                        zeno::vec3f background;
                        if ((h / gridSize) % 2 == (w / gridSize) % 2) {
                            background = {1, 1, 1};
                        }
                        else {
                            background = {0.86, 0.86, 0.86};
                        }
                        zeno::vec3f c = zeno::mix(background, foreground, alpha[i]);

                        int r = glm::clamp(int(c[0] * 255.99), 0, 255);
                        int g = glm::clamp(int(c[1] * 255.99), 0, 255);
                        int b = glm::clamp(int(c[2] * 255.99), 0, 255);

                        img.setPixel(w, height - 1 - h, qRgb(r, g, b));
                        //img.setPixel(width - 1 - w, h, qRgb(r, g, b));
                    }
                }
                else{
                    for (auto i = 0; i < obj->verts.size(); i++) {
                        int h = i / width;
                        int w = i % width;
                        auto c = obj->verts[i];
                        if (enableGamma) {
                            c = zeno::pow(c, 1.0f / 2.2f);
                        }
                        int r = glm::clamp(int(c[0] * 255.99), 0, 255);
                        int g = glm::clamp(int(c[1] * 255.99), 0, 255);
                        int b = glm::clamp(int(c[2] * 255.99), 0, 255);

                        img.setPixel(w, height - 1 - h, qRgb(r, g, b));
                    }
                }
                image_view->setImage(img);
            }
            QString statusInfo = QString(zeno::format("width: {}, height: {}", width, height).c_str());
            pStatusBar->setText(statusInfo);
        }
    }
    if (found == false) {
        clear();
    }
}

ZenoImagePanel::ZenoImagePanel(QWidget *parent) : QWidget(parent) {
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

    QLabel* pPrim = new QLabel(tr("Prim: "));
    pPrim->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pPrim);

    pPrimName->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pPrimName);

    pGamma->setStyleSheet("color: white;");
    pGamma->setCheckState(Qt::Checked);
    pTitleLayout->addWidget(pGamma);

    pAlpha->setStyleSheet("color: white;");
    pAlpha->setCheckState(Qt::Unchecked);
    pTitleLayout->addWidget(pAlpha);

    pFit->setProperty("cssClass", "grayButton");
    pTitleLayout->addWidget(pFit);

    pMainLayout->addLayout(pTitleLayout);

    image_view = new ZenoImageView(this);
    image_view->setProperty("cssClass", "proppanel");
    pMainLayout->addWidget(image_view);

    pStatusBar->setProperty("cssClass", "proppanel");
    pStatusBar->setText("PlaceHolder");


    pMainLayout->addWidget(pStatusBar);

    auto mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    QVector<DisplayWidget*> wids = mainWin->viewports();
    if (wids.isEmpty())
        return;

    Zenovis* zenovis = wids[0]->getZenoVis();
    if (!zenovis)
        return;

    connect(zenovis, &Zenovis::objectsUpdated, this, [=](int frame) {
        std::string prim_name = pPrimName->text().toStdString();
        Zenovis* zenovis = wids[0]->getZenoVis();
        ZASSERT_EXIT(zenovis);
        auto session = zenovis->getSession();
        ZASSERT_EXIT(session);
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
            if (key.find(prim_name) == 0 && key.find(zeno::format(":{}:", frame)) != std::string::npos) {
                setPrim(key);
            }
        }
    });
    connect(pGamma, &QCheckBox::stateChanged, this, [=](int state) {
        std::string prim_name = pPrimName->text().toStdString();
        Zenovis* zenovis = wids[0]->getZenoVis();
        ZASSERT_EXIT(zenovis);
        auto session = zenovis->getSession();
        ZASSERT_EXIT(session);
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
            if (key.find(prim_name) == 0) {
                setPrim(key);
            }
        }
    });
    connect(pAlpha, &QCheckBox::stateChanged, this, [=](int state) {
        std::string prim_name = pPrimName->text().toStdString();
        Zenovis* zenovis = wids[0]->getZenoVis();
        ZASSERT_EXIT(zenovis);
        auto session = zenovis->getSession();
        ZASSERT_EXIT(session);
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
            if (key.find(prim_name) == 0) {
                setPrim(key);
            }
        }
    });

    connect(pFit, &QPushButton::clicked, this, [=](bool _) {
        image_view->fitMode = true;
        image_view->updateImageView();
    });
    connect(image_view, &ZenoImageView::pixelChanged, this, [=](float x, float y) {
        std::string primid = pPrimName->text().toStdString();
        zenovis::Scene* scene = nullptr;
        auto mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin);
        QVector<DisplayWidget*> wids = mainWin->viewports();
        if (!wids.isEmpty())
        {
            auto session = wids[0]->getZenoVis()->getSession();
            ZASSERT_EXIT(session);
            scene = session->get_scene();
        }
        if (!scene)
            return;
        bool found = false;
        for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
            if ((key.substr(0, key.find(":"))) != primid) {
                continue;
            }
            auto &ud = ptr->userData();
            if (ud.get2<int>("isImage", 0) == 0) {
                continue;
            }
            found = true;
            if (auto obj = dynamic_cast<zeno::PrimitiveObject *>(ptr)) {
                int width = ud.get2<int>("w");
                int height = ud.get2<int>("h");
                int w = int(zeno::clamp(x, 0, width - 1));
                int h = int(zeno::clamp(y, 0, height - 1));
                int i = h * width + w;
                auto c = obj->verts[i];
                QString statusInfo = QString(zeno::format("width: {}, height: {} | ({}, {}) : ({}, {}, {})"
                        , width
                        , height
                        , w
                        , h
                        , c[0]
                        , c[1]
                        , c[2]
                        ).c_str());
                pStatusBar->setText(statusInfo);
            }
        }
        if (found == false) {
            clear();
        }
    });
}

