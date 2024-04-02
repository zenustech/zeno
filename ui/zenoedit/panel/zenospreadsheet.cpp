//
// Created by zh on 2022/6/27.
//

#include "zenospreadsheet.h"
#include "PrimAttrTableModel.h"
#include "viewport/zenovis.h"
#include "zenovis/ObjectsManager.h"
#include "zeno/utils/format.h"
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zenoui/comctrl/zcombobox.h>
#include "zenomainwindow.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "dialog/zforksubgrapdlg.h"
#include "nodesview/zenographseditor.h"
#include "settings/zenosettingsmanager.h"
#include <zenomodel/include/uihelper.h>

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

    m_checkSortingEnabled = new QCheckBox(this);
    m_checkSortingEnabled->setProperty("cssClass", "proppanel");
    m_checkSortingEnabled->setText(tr("enable sort"));
    pTitleLayout->addWidget(m_checkSortingEnabled);
    m_checkStringMapping = new QCheckBox(this);
    m_checkStringMapping->setProperty("cssClass", "proppanel");
    m_checkStringMapping->setText(tr("String mapping"));
    pTitleLayout->addWidget(m_checkStringMapping);


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

    prim_attr_view = new QTableView();
    prim_attr_view->setAlternatingRowColors(true);
    prim_attr_view->setSortingEnabled(false);
    prim_attr_view->setProperty("cssClass", "proppanel");
    prim_attr_view->setModel(sortModel);
    prim_attr_view->installEventFilter(this);
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
        for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
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
            prim_attr_view->setSortingEnabled(m_checkSortingEnabled->checkState());
        }
    });

    // enable sort
    connect(m_checkSortingEnabled, &QCheckBox::stateChanged, this, [this](int state) {
        prim_attr_view->setSortingEnabled(state != Qt::CheckState::Unchecked);
    });
    connect(m_checkStringMapping, &QCheckBox::stateChanged, this, [this](int state) {
        dataModel->setStrMapping(state != Qt::CheckState::Unchecked);
        prim_attr_view->update();
    });

    // corner button of tableview
    auto cornerBtn = prim_attr_view->findChild<QAbstractButton*>();
    // do not select all when clicked
    cornerBtn->disconnect();
    // reset sort order
    connect(cornerBtn, &QAbstractButton::clicked, this, [&, sortModel]() {
        sortModel->sort(-1);
        prim_attr_view->horizontalHeader()->setSortIndicator(-1, Qt::SortOrder::AscendingOrder);
    });

    connect(prim_attr_view, &QTableView::doubleClicked, this, [=](const QModelIndex& index) {
    QString label = prim_attr_view->model()->headerData(index.row(), Qt::Vertical).toString();
    if (label.contains("Material", Qt::CaseInsensitive))
    {
        QString mtlid = index.data(Qt::DisplayRole).toString();
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        if (pGraphsModel)
        {
            for (const auto& subgIdx : pGraphsModel->subgraphsIndice(SUBGRAPH_METERIAL))
            {
                if (subgIdx.data(ROLE_MTLID).toString() == mtlid)
                {
                    QString subgraph_name = subgIdx.data(ROLE_OBJNAME).toString();
                    ZenoMainWindow* pWin = zenoApp->getMainWindow();
                    if (pWin) {
                        ZenoSettingsManager::GetInstance().setValue(zsSubgraphType, SUBGRAPH_METERIAL);
                        ZenoGraphsEditor* pEditor = pWin->getAnyEditor();
                        if (pEditor)
                            pEditor->activateTab(subgraph_name, "", "");
                    }
                }
            }
        }
    }
    //QMimeData* pMimeData = new QMimeData;
    //pMimeData->setText(index.data(Qt::DisplayRole).toString());
    //QApplication::clipboard()->setMimeData(pMimeData);
    });
    connect(prim_attr_view->verticalHeader(), &QHeaderView::sectionDoubleClicked, this, [=](int index) {
        if (pMode->currentText() == "UserData")
        {
            auto graph_model = zenoApp->graphsManagment()->currentModel();
            if (!graph_model)
                return;
            const auto setType = [&](QModelIndex& node, QString type) {
                if (auto graph_model = zenoApp->graphsManagment()->currentModel()) {
                    QModelIndex& idx = node;
                    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
                    const QModelIndex& paramIdx = nodeParams->getParam(PARAM_INPUT, QString::fromStdString("data"));
                    graph_model->ModelSetData(paramIdx, type, ROLE_PARAM_TYPE);
                    graph_model->ModelSetData(paramIdx, UiHelper::getControlByType(type), ROLE_PARAM_CTRL);
                }
            };
            auto& node_sync = zeno::NodeSyncMgr::GetInstance();
            auto prim_node_location = node_sync.searchNodeOfPrim(pPrimName->text().toStdString());
            if (!prim_node_location.has_value())
                return;
            //auto out_sock = node_sync.getPrimSockName(prim_node_location.value());
            NODE_DATA ndata = graph_model->itemData(prim_node_location.value().node, graph_model->index("main"));
            std::string out_sock("");
            for (OUTPUT_SOCKET& outputSock: ndata[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>())
            {
                out_sock = outputSock.info.name.toStdString();
                break;
            }

            auto new_node_location = node_sync.generateNewNode(prim_node_location.value(),
                "SetUserData2",
                out_sock,
                "object");

            node_sync.updateNodeInputString(new_node_location.value(), "key", prim_attr_view->model()->headerData(index, Qt::Vertical).toString().toStdString());
            const QModelIndex& idx = prim_attr_view->model()->index(index, 0);
            const QStringList& vecLst = idx.data(Qt::DisplayRole).toString().split(",", QString::SkipEmptyParts);

            zeno::zany object = dataModel->userDataByIndex(idx);
            if (zeno::objectIsLiterial<float>(object)) {
                node_sync.updateNodeInputNumeric<float>(new_node_location.value(), "data", zeno::objectToLiterial<float>(object));
                setType(new_node_location.value().node, "float");
            }
            else if (zeno::objectIsLiterial<int>(object)) {
                node_sync.updateNodeInputNumeric<int>(new_node_location.value(), "data", zeno::objectToLiterial<int>(object));
                setType(new_node_location.value().node, "int");
            }
            else if (zeno::objectIsLiterial<zeno::vec2f>(object) || zeno::objectIsLiterial<zeno::vec2i>(object)) {
                node_sync.setNodeInputVec(new_node_location.value(), "data", UI_VECTYPE({ vecLst[0].toFloat(), vecLst[1].toFloat() }));
                setType(new_node_location.value().node, "vec2f");
            }
            else if (zeno::objectIsLiterial<zeno::vec3f>(object) || zeno::objectIsLiterial<zeno::vec3i>(object)) {
                node_sync.setNodeInputVec(new_node_location.value(), "data", UI_VECTYPE({vecLst[0].toFloat(), vecLst[1].toFloat(), vecLst[2].toFloat() }));
                setType(new_node_location.value().node, "vec3f");
            }
            else if (zeno::objectIsLiterial<zeno::vec4f>(object) || zeno::objectIsLiterial<zeno::vec4i>(object)) {
                node_sync.setNodeInputVec(new_node_location.value(), "data", UI_VECTYPE({vecLst[0].toFloat(), vecLst[1].toFloat(), vecLst[2].toFloat(), vecLst[3].toFloat() }));
                setType(new_node_location.value().node, "vec4f");
            }
            else if (zeno::objectIsLiterial<std::string>(object)) {
                node_sync.updateNodeInputString(new_node_location.value(), "data", idx.data(Qt::DisplayRole).toString().toStdString());
                setType(new_node_location.value().node, "string");
            }
            node_sync.updateNodeVisibility(new_node_location.value());
        }
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
    for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
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
            size_t num_uvs = obj->uvs.size();

            QString statusInfo = QString("Vertex: %1, Triangle: %2, Loops: %3, Poly: %4, Lines: %5, UserData: %6, Attribute: %7, UV: %8")
                .arg(num_vert)
                .arg(num_tris)
                .arg(num_loops)
                .arg(num_polys)
                .arg(num_lines)
                .arg(sizeUserData)
                .arg(num_attrs)
                .arg(num_uvs);
            pStatusBar->setText(statusInfo);
            this->dataModel->setModelData(obj);
        }
    }
    if (found == false) {
        this->dataModel->setModelData(nullptr);
    }

}

bool ZenoSpreadsheet::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == prim_attr_view && event->type() == QEvent::KeyPress)
    {
        QKeyEvent* keyEvt = dynamic_cast<QKeyEvent*>(event);
        if (keyEvt == QKeySequence::Copy) {
            if (QItemSelectionModel* pSelectionModel = prim_attr_view->selectionModel())
            {
                int cols = dataModel->columnCount(QModelIndex());
                const QModelIndexList& lst = pSelectionModel->selectedIndexes();
                QString copyStr;
                QList<int> rowsExist;
                for (const auto& idx : lst)
                {
                    int row = idx.row();
                    if (rowsExist.contains(row))
                        continue;
                    rowsExist << row;
                    QString str;
                    for (int col = 0; col < cols; col++)
                    {
                        const QModelIndex index = prim_attr_view->model()->index(row, col);
                        if (lst.contains(index))
                        {
                            if (!str.isEmpty())
                                str += ":";
                            str += index.data(Qt::DisplayRole).toString();
                        }

                    }
                    if (!str.isEmpty())
                    {
                        if (!copyStr.isEmpty())
                            copyStr += ",";
                        copyStr += str;
                    }
                }
                if (!copyStr.isEmpty())
                {
                    QMimeData* pMimeData = new QMimeData;
                    pMimeData->setText(copyStr);
                    QApplication::clipboard()->setMimeData(pMimeData);
                    return true;
                }
            }
        }
    }
    else if (watched == prim_attr_view && event->type() == QEvent::ContextMenu)
    {
        QStringList matLst;
        if (QItemSelectionModel* pSelectionModel = prim_attr_view->selectionModel())
        {
            const QModelIndexList& lst = pSelectionModel->selectedRows();
            for (auto index : lst)
            {
                int row = index.row();
                QString label = prim_attr_view->model()->headerData(row, Qt::Vertical).toString();
                if (label.contains("Material", Qt::CaseInsensitive))
                {
                    QString mtlid = index.data(Qt::DisplayRole).toString();
                    matLst << mtlid;
                }
            }
        }
        if (!matLst.isEmpty())
        {
            QMenu* pMenu = new QMenu;
            QAction* newSubGraph = new QAction(tr("Create Material Subgraph"));
            pMenu->addAction(newSubGraph);
            QAction* newMatSubGraph = new QAction(tr("Fork Preset Material Subgraphs"));
            pMenu->addAction(newMatSubGraph);
            QMenu* pPresetMenu = new QMenu(tr("Preset Material Subgraph"), pMenu);
            pMenu->addMenu(pPresetMenu);
            IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
            const auto& nodeIdx = pGraphsModel->nodeIndex(pPrimName->text());
            ZASSERT_EXIT(nodeIdx.isValid(), false);
            for (const auto& subgIdx : pGraphsModel->subgraphsIndice(SUBGRAPH_PRESET))
            {
                QString name = subgIdx.data(ROLE_OBJNAME).toString();
                QAction* pAction = new QAction(name);
                pPresetMenu->addAction(pAction);
                connect(pAction, &QAction::triggered, this, [=]() {
                    QMap<QString, QString> map;
                for (const auto& mat : matLst)
                {
                    map[mat] = pAction->text();
                }

                ZForkSubgraphDlg dlg(map, this);
                dlg.setNodeIdex(nodeIdx);
                dlg.exec();
                });
            }

            connect(newSubGraph, &QAction::triggered, this, [=]() {
                QPointF pos = nodeIdx.data(ROLE_OBJPOS).toPointF();
                for (const auto& mtlid : matLst)
                {
                    const auto& sugIdx = nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
                    if (!pGraphsModel->newMaterialSubgraph(sugIdx, mtlid, pos + QPointF(600, 0)))
                        QMessageBox::warning(nullptr, tr("Info"), tr("Create material subgraph '%1' failed.").arg(mtlid));
                }
            });
            connect(newMatSubGraph, &QAction::triggered, this, [=]() {
                getKeyWords();
                QMap<QString, QString> map;
                QString defaultMat;
                for (const auto& keywords : m_keyWords)
                {
                    if (keywords == "default")
                    {
                        defaultMat = m_keyWords.key(keywords);
                    }
                }
                for (const auto& mtlid : matLst)
                {
                    bool bFind = false;
                    for (const auto& keywords : m_keyWords)
                    {
                        if (keywords.isEmpty())
                            continue;
                        QRegularExpression re(keywords, QRegularExpression::CaseInsensitiveOption);
                        QRegularExpressionMatch match =  re.match(mtlid);
                        if (match.hasMatch())
                        {
                            map[mtlid] = m_keyWords.key(keywords);
                            bFind = true;
                            break;
                        }
                    }
                    if (!bFind)
                    {
                        if (!defaultMat.isEmpty())
                            map[mtlid] = defaultMat;
                        else
                            zeno::log_warn("can not match {}", mtlid.toStdString());
                    }
                    
                }
                ZForkSubgraphDlg dlg(map, this);
                dlg.setNodeIdex(nodeIdx);
                dlg.exec();
            });
            
            pMenu->exec(QCursor::pos());
            pMenu->deleteLater();
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ZenoSpreadsheet::getKeyWords()
{
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    const auto indexList = pGraphsModel->subgraphsIndice(SUBGRAPH_PRESET);
    if (indexList.isEmpty())
        return;
    QDialog dlg;
    dlg.setWindowTitle(tr("Set Key Words"));
    QVBoxLayout* pLayout = new QVBoxLayout(&dlg);

    QTableWidget* keyTableWidget = new QTableWidget(&dlg); 
    keyTableWidget->verticalHeader()->setVisible(false);
    //keyTableWidget->setProperty("cssClass", "select_subgraph");
    keyTableWidget->setColumnCount(2);
    QStringList labels = { tr("Preset Subgraph"), tr("key words") };
    keyTableWidget->setHorizontalHeaderLabels(labels);
    keyTableWidget->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    keyTableWidget->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    for (const auto& subgIdx : indexList)
    {
        int row = keyTableWidget->rowCount();
        keyTableWidget->insertRow(row);
        QString name = subgIdx.data(ROLE_OBJNAME).toString();
        QTableWidgetItem* pItem = new QTableWidgetItem(name);
        pItem->setFlags(pItem->flags() & ~Qt::ItemIsEditable);
        keyTableWidget->setItem(row, 0, pItem);

        QTableWidgetItem* pKeyItem = new QTableWidgetItem();
        keyTableWidget->setItem(row, 1, pKeyItem);
        if (m_keyWords.contains(name))
        {
            pKeyItem->setText(m_keyWords[name]);
        }
    }
    if (keyTableWidget->rowCount() > 0)
    {
        int height = keyTableWidget->rowHeight(0) * keyTableWidget->rowCount();
        int hearderH = keyTableWidget->horizontalHeader()->height();
        keyTableWidget->setMinimumHeight(height + hearderH);
    }
    keyTableWidget->viewport()->installEventFilter(this);
    QDialogButtonBox* pButtonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    QLabel* pLabel = new QLabel(tr("Separated by '|', such as : W1|W2|W3..."), &dlg);
    pLayout->addWidget(pLabel);
    pLayout->addWidget(keyTableWidget);
    pLayout->addWidget(pButtonBox);
    connect(pButtonBox, SIGNAL(accepted()), &dlg, SLOT(accept()));
    connect(pButtonBox, SIGNAL(rejected()), &dlg, SLOT(reject()));
    if (QDialog::Accepted == dlg.exec()) {
        for (int row = 0; row < keyTableWidget->rowCount(); row++)
        {
            QString keys = keyTableWidget->item(row, 1)->text();
            if (!keys.isEmpty())
            {
                QString jsonKey = keyTableWidget->item(row, 0)->text();
                 m_keyWords[jsonKey] = keys;
            }
        }
    }
}