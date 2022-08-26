//
// Created by zh on 2022/8/26.
//

#include "ZenoObjectList.h"
#include "viewport/zenovis.h"

ZenoObjectList::ZenoObjectList(QWidget *parent) {
    QVBoxLayout* pMainLayout = new QVBoxLayout;
    pMainLayout->setContentsMargins(QMargins(0, 0, 0, 0));
    setLayout(pMainLayout);
    setFocusPolicy(Qt::ClickFocus);

    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(37, 37, 38));
    setPalette(palette);
    setAutoFillBackground(true);

    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);


    lights_view->setAlternatingRowColors(true);
    lights_view->setProperty("cssClass", "proppanel");
    lights_view->setModel(this->dataModel);
    pMainLayout->addWidget(lights_view);

    pStatusBar->setProperty("cssClass", "proppanel");
    pMainLayout->addWidget(pStatusBar);

    auto * zenovis = &Zenovis::GetInstance();
    connect(zenovis, &Zenovis::objectsUpdated, this, [&](int frame) {
        this->dataModel->updateByObjectsMan();
    });
}
