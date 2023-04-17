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
    pPrimName->setText(QString(primid.c_str()).split(':')[0]);
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    bool found = false;
    for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
        if (key != primid) {
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
                for (auto i = 0; i < obj->verts.size(); i++) {
                    int h = i / width;
                    int w = i % width;
                    auto c = obj->verts[i];
                    int r = glm::clamp(int(c[0] * 255.99), 0, 255);
                    int g = glm::clamp(int(c[1] * 255.99), 0, 255);
                    int b = glm::clamp(int(c[2] * 255.99), 0, 255);

                    img.setPixel(w, height - 1 - h, qRgb(r, g, b));
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
    auto *zenovis = &Zenovis::GetInstance();
    connect(zenovis, &Zenovis::objectsUpdated, this, [&](int frame) {
        std::string prim_name = pPrimName->text().toStdString();
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
            if (key.find(prim_name) == 0 && key.find(zeno::format(":{}:", frame)) != std::string::npos) {
                setPrim(key);
            }
        }
    });
}
