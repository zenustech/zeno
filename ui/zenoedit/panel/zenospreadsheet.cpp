//
// Created by zh on 2022/6/27.
//

#include "zenospreadsheet.h"
#include "PrimAttrTableModel.h"
#include "viewport/zenovis.h"
#include "zeno/extra/ObjectsManager.h"
#include "zeno/utils/format.h"
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zenoui/comctrl/zcombobox.h>
#include "zenomainwindow.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalComm.h>


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


    ZComboBox* pMode = new ZComboBox();
    pMode->addItem("Vertex");
    pMode->addItem("Tris");
    pMode->addItem("Points");
    pMode->addItem("Lines");
    pMode->addItem("Quads");
    pMode->addItem("Polys");
    pMode->addItem("Loops");
    pMode->addItem("UVs");
    pMode->addItem("UserData");
    pMode->setProperty("cssClass", "proppanel");
    pTitleLayout->addWidget(pMode);

    pMainLayout->addLayout(pTitleLayout);

    auto sortModel = new QSortFilterProxyModel(this);
    sortModel->setSourceModel(dataModel);

    QTableView *prim_attr_view = new QTableView();
    prim_attr_view->setAlternatingRowColors(true);
    prim_attr_view->setSortingEnabled(true);
    prim_attr_view->setProperty("cssClass", "proppanel");
    prim_attr_view->setModel(sortModel);
    pMainLayout->addWidget(prim_attr_view);

//    pStatusBar->setAlignment(Qt::AlignRight);
    pStatusBar->setProperty("cssClass", "proppanel");
    pMainLayout->addWidget(pStatusBar);

    ZenoMainWindow *pWin = zenoApp->getMainWindow();
    ZERROR_EXIT(pWin);

    connect(pWin, &ZenoMainWindow::visObjectsUpdated, this, [=](ViewportWidget* pViewport, int frame) {
        ZERROR_EXIT(pViewport);
        auto zenovis = pViewport->getZenoVis();
        ZERROR_EXIT(zenovis);

        std::string prim_name = pPrimName->text().toStdString();
        auto sess = zenovis->getSession();
        ZERROR_EXIT(sess);
        auto scene = zenovis->getSession()->get_scene();
        ZERROR_EXIT(scene);
        for (auto const &[key, ptr]: zeno::getSession().globalComm->objectsMan->pairs()) {
            if (key.find(prim_name) == 0 && key.find(zeno::format(":{}:", frame)) != std::string::npos) {
                setPrim(key);
            }
        }
    });

    connect(pMode, &ZComboBox::_textActivated, [=](const QString& text) {
        this->dataModel->setSelAttr(text.toStdString());
        // do not sort for userdata
        if (text == "UserData") {
            // reset sort order
            sortModel->sort(-1);
            prim_attr_view->horizontalHeader()->setSortIndicator(-1, Qt::SortOrder::AscendingOrder);
            prim_attr_view->setSortingEnabled(false);
        }
        else {
            prim_attr_view->setSortingEnabled(true);
        }
    });

    // corner button of tableview
    auto cornerBtn = prim_attr_view->findChild<QAbstractButton*>();
    // do not select all when clicked
    cornerBtn->disconnect();
    // reset sort order
    connect(cornerBtn, &QAbstractButton::clicked, this, [sortModel, prim_attr_view]() {
        sortModel->sort(-1);
        prim_attr_view->horizontalHeader()->setSortIndicator(-1, Qt::SortOrder::AscendingOrder);
    });
}

void ZenoSpreadsheet::clear() {
    pPrimName->clear();
    this->dataModel->setModelData(nullptr);
    pStatusBar->clear();
}

void ZenoSpreadsheet::setPrim(std::string primid) {
    pPrimName->setText(QString(primid.c_str()).split(':')[0]);

    ZenoMainWindow* pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);

    QVector<DisplayWidget*> views = zenoApp->getMainWindow()->viewports();
    if (views.isEmpty())
        return;

    ZASSERT_EXIT(views[0]);
    Zenovis* pZenovis = views[0]->getZenoVis();
    ZASSERT_EXIT(pZenovis);
    auto scene = pZenovis->getSession()->get_scene();
    ZASSERT_EXIT(scene);

    bool found = false;
    for (auto const &[key, ptr]: zeno::getSession().globalComm->objectsMan->pairs()) {
        if (key != primid) {
            continue;
        }
        if (auto obj = dynamic_cast<zeno::PrimitiveObject *>(ptr)) {
            found = true;
            size_t sizeUserData = obj->userData().size();
            size_t num_attrs = obj->num_attrs();
            size_t num_vert = obj->verts.size();
            size_t num_tris = obj->tris.size();
            size_t num_loops = obj->loops.size();
            size_t num_polys = obj->polys.size();
            size_t num_lines = obj->lines.size();

            QString statusInfo = QString("Vertex: %1, Triangle: %2, Loops: %3, Poly: %4, Lines: %5, UserData: %6, Attribute: %7")
                .arg(num_vert)
                .arg(num_tris)
                .arg(num_loops)
                .arg(num_polys)
                .arg(num_lines)
                .arg(sizeUserData)
                .arg(num_attrs);
            pStatusBar->setText(statusInfo);
            this->dataModel->setModelData(obj);
        }
    }
    if (found == false) {
        this->dataModel->setModelData(nullptr);
    }

}
