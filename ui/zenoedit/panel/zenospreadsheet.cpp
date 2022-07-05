//
// Created by zh on 2022/6/27.
//

#include "zenospreadsheet.h"
#include "PrimAttrTableModel.h"
#include "viewport/zenovis.h"
#include "zenovis/ObjectsManager.h"
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>

ZenoSpreadsheet::ZenoSpreadsheet(QWidget *parent) : QWidget(parent) {
    dataModel = new PrimAttrTableModel();
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


    QComboBox* pMode = new QComboBox();
    pMode->addItem("Vertex");
    pMode->addItem("Face");
    pMode->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pMode);

    QLabel* pAttribute = new QLabel("ATTRIBUTE");
    pAttribute->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pAttribute);

    QComboBox* pAttributeName = new QComboBox();
    pAttributeName->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pAttributeName);

    pMainLayout->addLayout(pTitleLayout);

    QTableView *prim_attr_view = new QTableView();
    prim_attr_view->setProperty("cssClass", "proppanel");
    prim_attr_view->setModel(dataModel);
    pMainLayout->addWidget(prim_attr_view);

//    pStatusBar->setAlignment(Qt::AlignRight);
    pStatusBar->setProperty("cssClass", "proppanel");
    pMainLayout->addWidget(pStatusBar);
}

void ZenoSpreadsheet::clear() {
    pPrimName->clear();
    pStatusBar->clear();
}

void ZenoSpreadsheet::setPrim(std::string primid) {
    pPrimName->setText(QString(primid.c_str()).split(':')[0]);
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    bool found = false;
    for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
        if (key != primid) {
            continue;
        }
        if (auto obj = dynamic_cast<zeno::PrimitiveObject *>(ptr)) {
            found = true;
            size_t sizeUserData = obj->userData().size();
            size_t num_attrs = obj->num_attrs();

            QString statusInfo = QString("UserData count: %1, Attribute count: %2").arg(sizeUserData).arg(num_attrs);
            pStatusBar->setText(statusInfo);
            this->dataModel->setModelData(obj);
        }
    }
    if (found == false) {
        this->dataModel->setModelData(nullptr);
    }

}
