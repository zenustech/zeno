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

const float ziv_wheelZoomFactor = 1.25;

class ZenoImageView: QGraphicsView {
    QGraphicsPixmapItem *_image = nullptr;
    QGraphicsScene *scene;
public:
    explicit ZenoImageView(QWidget *parent) : QGraphicsView(parent) {
        scene = new QGraphicsScene;
        this->setScene(scene);

        this->setHorizontalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
        this->setVerticalScrollBarPolicy(Qt::ScrollBarPolicy::ScrollBarAlwaysOff);
    }

    bool hasImage() {
        return _image != nullptr;
    }

    void clearImage() {
        if (_image) {

        }
    }

    void setImage(QImage image) {
        QPixmap pm = QPixmap::fromImage(image);
        clearImage();
        _image = this->scene->addPixmap(pm);

    }
};



void ZenoImagePanel::clear() {
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
        if (ud.get2<int>("image", 0) == 0) {
            continue;
        }
        found = true;
        if (auto obj = dynamic_cast<zeno::PrimitiveObject *>(ptr)) {
            QString statusInfo = QString("Placeholder");
            pStatusBar->setText(statusInfo);
        }
    }
    if (found == false) {

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

    ZComboBox* pMode = new ZComboBox();
    pMode->addItem("RGB");
    pMode->addItem("RGBA");
    pMode->addItem("R");
    pMode->addItem("G");
    pMode->addItem("B");
    pMode->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pMode);

    pMainLayout->addLayout(pTitleLayout);

    QGraphicsView *image_view = new QGraphicsView();
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
