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


const float ziv_wheelZoomFactor = 1.25;

class ZenoImageView: public QGraphicsView {
public:
    QGraphicsPixmapItem *_image = nullptr;
    QGraphicsScene *scene = nullptr;
    explicit ZenoImageView(QWidget *parent) : QGraphicsView(parent) {
        scene = new QGraphicsScene;
        this->setScene(scene);

        setBackgroundBrush(QColor(37, 37, 37));

        this->setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
        this->setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
    }

    bool hasImage() {
        return _image != nullptr;
    }

    void clearImage() {
        if (hasImage()) {
            scene->removeItem(_image);
            _image = nullptr;
        }
    }

    void setImage(const QImage &image) {
        QPixmap pm = QPixmap::fromImage(image);
        if (hasImage()) {
            _image->setPixmap(pm);
        }
        else {
            _image = this->scene->addPixmap(pm);
        }
        setSceneRect(QRectF(pm.rect()));  // Set scene size to image size.
        updateImageView();
    }

    void updateImageView() {
        if (!hasImage()) {
            return;
        }

        fitInView(sceneRect(), Qt::AspectRatioMode::KeepAspectRatio);
    }
    void resizeEvent(QResizeEvent *event) override {
        updateImageView();
    }
};

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
                std::vector<zeno::vec3f> rgb;
                if (obj->verts.has_attr("alpha")) {
                    for (int i = 0; i < height; ++i) {
                        for (int j = 0; j < width; ++j) {
                            if(obj->verts.attr<float>("alpha")[i * width + j]==0){
                                if ((i / gridSize) % 2 == (j / gridSize) % 2) {
                                    rgb.push_back(zeno::vec3f(220,220,220));
                                } else {
                                    rgb.push_back(zeno::vec3f(255, 255, 255));
//                                    img.setPixel(j, height-1-i, qRgba(255, 255, 255, 255)); // 白色格子
                                }
                            }
                            else if(obj->verts.attr<float>("alpha")[i * width + j]!=0 && obj->verts.attr<float>("alpha")[i * width + j]!=1){
                                auto a1 = obj->verts.attr<float>("alpha")[i * width + j];
                                auto c = obj->verts[i * width + j];
                                if ((i / gridSize) % 2 == (j / gridSize) % 2) {
                                    c = zeno::pow(c, 1.0f / 2.2f);
                                    rgb.push_back(zeno::vec3f(glm::clamp(int(c[0] * 255.99 * a1 + 220 * (1-a1)), 0, 255),
                                                              glm::clamp(int(c[1] * 255.99 * a1 + 220 * (1-a1)), 0, 255),
                                                              glm::clamp(int(c[2] * 255.99 * a1 + 220 * (1-a1)), 0, 255)));
                                } else {
                                    c = zeno::pow(c, 1.0f / 2.2f);
                                    rgb.push_back(zeno::vec3f(glm::clamp(int(c[0] * 255.99 * a1 + 255 * (1-a1)), 0, 255),
                                                              glm::clamp(int(c[1] * 255.99 * a1 + 255 * (1-a1)), 0, 255),
                                                              glm::clamp(int(c[2] * 255.99 * a1 + 255 * (1-a1)), 0, 255)));
                                }
                            }
                            else{
                                auto c = obj->verts[i * width + j];
                                c = zeno::pow(c, 1.0f / 2.2f);
                                rgb.push_back(zeno::vec3f(glm::clamp(int(c[0] * 255.99), 0, 255),
                                                          glm::clamp(int(c[1] * 255.99), 0, 255),
                                                          glm::clamp(int(c[2] * 255.99), 0, 255)));
                            }
                        }
                    }
                }
                else{
                    for (auto i = 0; i < obj->verts.size(); i++) {
                        auto c = obj->verts[i];
                        c = zeno::pow(c, 1.0f / 2.2f);
                        rgb.push_back(zeno::vec3f(glm::clamp(int(c[0] * 255.99), 0, 255),
                                                  glm::clamp(int(c[1] * 255.99), 0, 255),
                                                  glm::clamp(int(c[2] * 255.99), 0, 255)));
                    }
                }
                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        img.setPixel(j, height - 1 - i, qRgba(rgb[i * width + j][0], rgb[i * width + j][1], rgb[i * width + j][2] ,255));
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

//    ZComboBox* pMode = new ZComboBox();
//    pMode->addItem("RGB");
//    pMode->addItem("RGBA");
//    pMode->addItem("R");
//    pMode->addItem("G");
//    pMode->addItem("B");
//    pMode->setProperty("cssClass", "proppanel");
//    pTitleLayout->addWidget(pMode);

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
}

