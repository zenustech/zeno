#include "docktabcontent.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zicontoolbutton.h>
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zstyleoption.h>
#include "../panel/zenodatapanel.h"
#include "../panel/zenoproppanel.h"
#include "../panel/zenospreadsheet.h"
#include "../panel/zlogpanel.h"
#include <zenoedit/panel/zenoimagepanel.h>
#include "nodesview/zenographseditor.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/uihelper.h>
#include <zenoui/comctrl/zlinewidget.h>
#include <zenoui/comctrl/view/zcomboboxitemdelegate.h>
#include <zenoui/comctrl/zwidgetfactory.h>
#include <zeno/utils/envconfig.h>
#include "zenomainwindow.h"
#include "launch/corelaunch.h"
#include "settings/zenosettingsmanager.h"
#include "settings/zsettings.h"
#include <zenoui/comctrl/zcombobox.h>
#include <zeno/core/Session.h>
#include <zeno/types/UserData.h>
#include <zenovis/ObjectsManager.h>


ZToolBarButton::ZToolBarButton(bool bCheckable, const QString& icon, const QString& iconOn)
    : ZToolButton(ZToolButton::Opt_HasIcon, icon, iconOn)
{
    setCheckable(bCheckable);

    QColor bgOn("#4F5963");

    int marginLeft = ZenoStyle::dpiScaled(0);
    int marginRight = ZenoStyle::dpiScaled(0);
    int marginTop = ZenoStyle::dpiScaled(0);
    int marginBottom = ZenoStyle::dpiScaled(0);

    setIcon(ZenoStyle::dpiScaledSize(QSize(20, 20)), icon, iconOn, iconOn, iconOn);

    setMargins(QMargins(marginLeft, marginTop, marginRight, marginBottom));
    setRadius(ZenoStyle::dpiScaled(2));
    setBackgroundClr(QColor(), bgOn, bgOn, bgOn);
}


#if 0
ZToolRecordingButton::ZToolRecordingButton(const QString &icon, const QString &iconHover, const QString &iconOn,const QString &iconOnHover, const QString &iconPressed)
    : ZToolButton()
{
    setButtonOptions(ZToolButton::Opt_TextLeftToIcon | ZToolButton::Opt_Checkable);
    setIcon(ZenoStyle::dpiScaledSize(QSize(24, 24)), icon, iconHover, iconOn, iconOnHover);
    QFont fnt = QApplication::font();
    setText(tr("REC"));
    setMargins(ZenoStyle::dpiScaledMargins(QMargins(12, 5, 5, 5)));
    setBackgroundClr(QColor("#383F47"), QColor("#383F47"), QColor("#191D21"), QColor("#191D21"));
    setTextClr(QColor(), QColor(), QColor("#FFFFFF"), QColor("#FFFFFF"));
    m_iconOnPressed = QIcon(iconPressed);
}

void ZToolRecordingButton::paintEvent(QPaintEvent *event)
{
    QStylePainter p(this);
    ZStyleOptionToolButton option;
    option.initFrom(this);
    if (!isChecked() && !isHovered())
    {
        option.icon = icon();
        option.text = "";
    }else if (!isChecked() && isHovered())
    {
        option.icon = icon();
        option.text = "";
    }else if (isChecked() && !isHovered() && !isPressed())
    {
        option.icon = icon();
        option.text = tr("REC");
        option.palette.setBrush(QPalette::All, QPalette::WindowText, QColor("#FFFFFF"));
    }else if (isChecked() && isHovered() && !isPressed())
    {
        option.icon = icon();
        option.text = tr("OFF");
        option.palette.setBrush(QPalette::All, QPalette::WindowText, QColor("#A3B1C0"));
    }else if (isChecked() && isHovered() && isPressed())
    {
        option.icon = m_iconOnPressed;
        option.text = tr("OFF");
        option.palette.setBrush(QPalette::All, QPalette::WindowText, QColor("#C3D2DF"));
    }
    option.iconSize = iconSize();
    option.buttonOpts = buttonOption();
    option.font = QApplication::font();
    option.bgRadius = ZenoStyle::dpiScaled(2);
    option.palette.setBrush(QPalette::All, QPalette::Window, QBrush(backgrondColor(option.state)));
    p.drawComplexControl(QStyle::CC_ToolButton, option);
}
#endif


ZToolMenuButton::ZToolMenuButton() {
    setButtonOptions(ZToolButton::Opt_TextRightToIcon);
    setArrowOption(ZStyleOptionToolButton::DOWNARROW);
    menu = new QMenu(this);
    run = new QAction(icon(), tr("Run"), this);
    runLightCamera = new QAction(icon(), tr("RunLightCamera"), this);
    runMaterial = new QAction(icon(), tr("RunMaterial"), this);
    connect(run, &QAction::triggered, this, [=]() {
        setText(tr("Run"));
        this->setMinimumWidth(sizeHint().width());
    });
    connect(runLightCamera, &QAction::triggered, this, [=]() {
        setText(tr("RunLightCamera"));
        this->setMinimumWidth(sizeHint().width());
    });
    connect(runMaterial, &QAction::triggered, this, [=]() {
        setText(tr("RunMaterial"));
        this->setMinimumWidth(sizeHint().width());
    });
    menu->addAction(run);
    menu->addAction(runLightCamera);
    menu->addAction(runMaterial);
}


void ZToolMenuButton::mouseReleaseEvent(QMouseEvent *e) {
    QSize size = ZToolButton::sizeHint();
    if (e->x() >= (size.width() - ZenoStyle::dpiScaled(10)))
    {
        QPoint pos;
        pos.setY(pos.y() + this->geometry().height());
        menu->exec(this->mapToGlobal(pos));
        return;
    }
    emit clicked();
}


QSize ZToolMenuButton::sizeHint() const {
    QSize size = ZToolButton::sizeHint();
    size.setWidth(size.width() + ZenoStyle::dpiScaled(12));
    return size;
}

const int DockToolbarWidget::sToolbarHeight = 28;


DockToolbarWidget::DockToolbarWidget(QWidget* parent)
    : QWidget(parent)
    , m_pWidget(nullptr)
{
}

void DockToolbarWidget::initUI()
{
    QVBoxLayout *pLayout = new QVBoxLayout;
    pLayout->setSpacing(0);
    pLayout->setContentsMargins(0, 0, 0, 0);

    QWidget *pToolbar = new QWidget;
    pToolbar->setFixedHeight(ZenoStyle::dpiScaled(sToolbarHeight));

    QHBoxLayout* pToolLayout = new QHBoxLayout;
    pToolLayout->setContentsMargins(ZenoStyle::dpiScaled(8), ZenoStyle::dpiScaled(4),
        ZenoStyle::dpiScaled(4), ZenoStyle::dpiScaled(4));
    pToolbar->setLayout(pToolLayout);

    initToolbar(pToolLayout);
    pLayout->addWidget(pToolbar);
    pLayout->addWidget(new ZPlainLine(1, QColor("#000000"))); //add the seperator line
    pLayout->addWidget(initWidget());

    initConnections();

    setLayout(pLayout);
}

void DockToolbarWidget::onTabAboutToClose()
{
}

QWidget* DockToolbarWidget::widget() const
{
    return m_pWidget;
}



DockContent_Parameter::DockContent_Parameter(QWidget* parent)
    : DockToolbarWidget(parent)
    , m_pSettingBtn(nullptr)
{
}

void DockContent_Parameter::initToolbar(QHBoxLayout* pToolLayout)
{
    //pToolLayout->setSpacing(9);

    ZIconLabel* pIcon = new ZIconLabel();
    pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(20, 20)), ":/icons/nodeclr-yellow.svg", "");

    m_plblName = new QLabel("");
    QFont fnt = QApplication::font();
    m_plblName->setFont(fnt);
    m_plblName->setTextInteractionFlags(Qt::TextSelectableByMouse);
    QPalette palette = m_plblName->palette();
    palette.setColor(m_plblName->foregroundRole(), QColor("#A3B1C0"));
    m_plblName->setPalette(palette);

    m_pNameLineEdit = new ZLineEdit;
    m_pNameLineEdit->setProperty("cssClass", "zeno2_2_lineedit");

    ZToolBarButton* pFixBtn = new ZToolBarButton(false, ":/icons/fixpanel.svg", ":/icons/fixpanel-on.svg");
    ZToolBarButton* pWikiBtn = new ZToolBarButton(false, ":/icons/wiki.svg", ":/icons/wiki-on.svg");
    m_pSettingBtn = new ZToolBarButton(false, ":/icons/settings.svg", ":/icons/settings-on.svg");

    pToolLayout->addWidget(pIcon);
    pToolLayout->addWidget(m_plblName);
    pToolLayout->addWidget(m_pNameLineEdit);
    pToolLayout->addStretch();
    pToolLayout->addWidget(pFixBtn);
    pToolLayout->addWidget(pWikiBtn);
    pToolLayout->addWidget(m_pSettingBtn);
}

QWidget* DockContent_Parameter::initWidget()
{
    m_pWidget = new ZenoPropPanel;
    return m_pWidget;
}

void DockContent_Parameter::initConnections()
{
    ZenoPropPanel* prop = qobject_cast<ZenoPropPanel*>(m_pWidget);
    connect(m_pSettingBtn, &ZToolBarButton::clicked, prop, &ZenoPropPanel::onSettings);

    connect(m_pNameLineEdit, &ZLineEdit::textEditFinished, this, [=]() {
        QString value = m_pNameLineEdit->text();
        QString oldValue;
        if (!prop->updateCustomName(value, oldValue)) 
        {
            m_pNameLineEdit->setText(oldValue);
        }
    });
}

void DockContent_Parameter::onNodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select)
{
    if (ZenoPropPanel* prop = findChild<ZenoPropPanel*>())
    {
        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        prop->reset(pModel, subgIdx, nodes, select);

        if (!nodes.isEmpty())
        {
            const QModelIndex& idx = nodes[0];
            const QAbstractItemModel* pSubgModel = idx.model();
            if (pSubgModel)
            {
                connect(pSubgModel, &QAbstractItemModel::dataChanged, this, [=](const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles) {
                    if (roles.isEmpty())
                    return;
                    int role = roles[0];
                    if (role != ROLE_CUSTOM_OBJNAME)
                        return;
                    m_pNameLineEdit->setText(idx.data(ROLE_CUSTOM_OBJNAME).toString());
                });
            }

            if (select) {
                m_plblName->setText(idx.data(ROLE_OBJID).toString());
                m_pNameLineEdit->setText(idx.data(ROLE_CUSTOM_OBJNAME).toString());
                return;
            }
        }
        m_plblName->setText("");
        m_pNameLineEdit->setText("");
    }
}

void DockContent_Parameter::onPrimitiveSelected(const std::unordered_set<std::string>& primids)
{

}


DockContent_Editor::DockContent_Editor(QWidget* parent)
    : DockToolbarWidget(parent)
    , m_pEditor(nullptr)
    , m_btnRun(nullptr)
    , m_btnKill(nullptr)
    , m_btnAlways(nullptr)
{
}

void DockContent_Editor::initToolbar(QHBoxLayout* pToolLayout)
{
    pListView = new ZToolBarButton(true, ":/icons/subnet-listview.svg", ":/icons/subnet-listview-on.svg");
    pTreeView = new ZToolBarButton(true, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    pSubnetMgr = new ZToolBarButton(false, ":/icons/nodeEditor_subnetManager_unselected.svg", ":/icons/nodeEditor_subnetManager_selected.svg");
    pFold = new ZToolBarButton(false, ":/icons/nodeEditor_nodeFold_unselected.svg", ":/icons/nodeEditor_nodeFold_selected.svg");
    pUnfold = new ZToolBarButton(false, ":/icons/nodeEditor_nodeUnfold_unselected.svg", ":/icons/nodeEditor_nodeUnfold_selected.svg");
    pSnapGrid = new ZToolBarButton(true, ":/icons/nodeEditor_snap_unselected.svg", ":/icons/nodeEditor_snap_selected.svg");
    pShowGrid = new ZToolBarButton(true, ":/icons/nodeEditor_grid_unselected.svg", ":/icons/nodeEditor_grid_selected.svg");
    pCustomParam = new ZToolBarButton(false, ":/icons/nodeEditor_nodePara_unselected.svg", ":/icons/nodeEditor_nodePara_selected.svg");
    pGroup = new ZToolBarButton(false, ":/icons/nodeEditor_blackboard_unselected.svg", ":/icons/nodeEditor_blackboard_selected.svg");
    pSearchBtn = new ZToolBarButton(true, ":/icons/toolbar_search_idle.svg", ":/icons/toolbar_search_light.svg");
    pSettings = new ZToolBarButton(false, ":/icons/toolbar_localSetting_idle.svg", ":/icons/toolbar_localSetting_light.svg");
    pLinkLineShape = new ZToolBarButton(true, ":/icons/timeline-curvemap.svg",":/icons/timeline-curvemap.svg");

    m_btnRun = new ZToolMenuButton;
    m_btnKill = new ZToolButton;

    QFont fnt = QApplication::font();

    m_btnRun->setIcon(ZenoStyle::dpiScaledSize(QSize(14, 14)), ":/icons/timeline_run_thunder.svg",
                          ":/icons/timeline_run_thunder.svg", "", "");
    m_btnRun->setRadius(ZenoStyle::dpiScaled(2));
    m_btnRun->setFont(fnt);
    m_btnRun->setText(tr("Run"));
    m_btnRun->setCursor(QCursor(Qt::PointingHandCursor));
    m_btnRun->setMargins(ZenoStyle::dpiScaledMargins(QMargins(11, 5, 14, 5)));
    m_btnRun->setBackgroundClr(QColor("#4578AC"), QColor("#4578AC"), QColor("#4578AC"), QColor("#4578AC"));
    m_btnRun->setTextClr(QColor("#FFFFFF"), QColor("#FFFFFF"), QColor("#FFFFFF"), QColor("#FFFFFF"));
    ZenoSettingsManager &settings = ZenoSettingsManager::GetInstance();
    m_btnRun->setShortcut(settings.getShortCut(ShortCut_Run));
    m_btnRun->setCursor(QCursor(Qt::PointingHandCursor));

    //kill
    m_btnKill->setButtonOptions(ZToolButton::Opt_TextRightToIcon);
    m_btnKill->setIcon(ZenoStyle::dpiScaledSize(QSize(14, 14)), ":/icons/timeline_kill_clean.svg",
                           ":/icons/timeline_kill_clean.svg", "", "");
    m_btnKill->setRadius(ZenoStyle::dpiScaled(2));
    m_btnKill->setFont(fnt);
    m_btnKill->setText(tr("Kill"));
    m_btnKill->setCursor(QCursor(Qt::PointingHandCursor));
    m_btnKill->setMargins(ZenoStyle::dpiScaledMargins(QMargins(11, 5, 14, 5)));
    m_btnKill->setBackgroundClr(QColor("#4D5561"), QColor("#4D5561"), QColor("#4D5561"), QColor("#4D5561"));
    m_btnKill->setTextClr(QColor("#FFFFFF"), QColor("#FFFFFF"), QColor("#FFFFFF"), QColor("#FFFFFF"));
    m_btnKill->setShortcut(settings.getShortCut(ShortCut_Kill));
    m_btnKill->setCursor(QCursor(Qt::PointingHandCursor));

    QStringList runList{tr("disable"), tr("alwaysAll"), tr("alwaysLightCamera"), tr("alwaysMaterial")};
    m_btnAlways = new ZComboBox(this);
    m_btnAlways->addItems(runList);
    m_btnAlways->setEditable(false);
    m_btnAlways->setFixedHeight(ZenoStyle::dpiScaled(22));
    m_btnAlways->setFont(fnt);
    m_btnAlways->setProperty("focusBorder", "none");
    QFontMetrics fontMetrics(fnt);
    m_btnAlways->view()->setMinimumWidth(fontMetrics.horizontalAdvance(tr("alwaysLightCamera")) + ZenoStyle::dpiScaled(30));
    QObject::connect(m_btnAlways, &ZComboBox::_textActivated, [=](const QString &text) {
        std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
        ZASSERT_EXIT(mgr);
        ZenoMainWindow *pMainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(pMainWin);
        std::function<void()> resetAlways = [=]() {
            m_btnAlways->setCurrentText(tr("disable"));
            pMainWin->setAlwaysLightCameraMaterial(false, false);
        };
        QVector<DisplayWidget *> views = pMainWin->viewports();
        for (auto displayWid : views) {
            if (!displayWid->isGLViewport()) {
                displayWid->setRenderSeparately(false, false);
            }
        }
        connect(zenoApp->graphsManagment(), &GraphsManagment::fileOpened, this, resetAlways);
        connect(zenoApp->graphsManagment(), &GraphsManagment::modelInited, this, resetAlways);
        if (text == tr("alwaysAll"))
        {
            mgr->setCacheOpt(ZCacheMgr::Opt_AlwaysOnAll);
            pMainWin->setAlways(true);
            pMainWin->setAlwaysLightCameraMaterial(false, false);
            pMainWin->onRunTriggered();
        }
        else if (text == tr("alwaysLightCamera") || text == tr("alwaysMaterial")) {
            QSettings settings(zsCompanyName, zsEditor);
            if (!settings.value("zencache-enable").toBool()) {
                QMessageBox::warning(nullptr, text, tr("This function can only be used in cache mode."));
                m_btnAlways->setCurrentIndex(0);
            } else {
                if (text == tr("alwaysLightCamera")) {
                    pMainWin->setAlwaysLightCameraMaterial(true, false);
                } else if (text == tr("alwaysMaterial")) {
                    pMainWin->setAlwaysLightCameraMaterial(false, true);
                }
                mgr->setCacheOpt(ZCacheMgr::Opt_AlwaysOnAll);
                pMainWin->setAlways(false);
                pMainWin->onRunTriggered();
            }
        }
        else {
            mgr->setCacheOpt(ZCacheMgr::Opt_Undefined);
            pMainWin->setAlways(false);
            pMainWin->setAlwaysLightCameraMaterial(false, false);
        }
        m_btnAlways->setFixedWidth(fontMetrics.horizontalAdvance(text) + ZenoStyle::dpiScaled(26));
    });
    m_btnAlways->setFixedWidth(fontMetrics.horizontalAdvance(tr("disable")) + ZenoStyle::dpiScaled(26));

    pListView->setChecked(false);
    pShowGrid->setChecked(ZenoSettingsManager::GetInstance().getValue(zsShowGrid).toBool());
    pSnapGrid->setChecked(ZenoSettingsManager::GetInstance().getValue(zsSnapGrid).toBool());
    pLinkLineShape->setChecked(ZenoSettingsManager::GetInstance().getValue(zsLinkLineShape).toBool());

    QStringList items;
    QVector<qreal> factors = UiHelper::scaleFactors();
    for (qreal factor : factors) {
        int per = factor * 100;
        QString sPer = QString("%1%").arg(per);
        items.append(sPer);
    }
    QVariant props = items;

    Callback_EditFinished funcZoomEdited = [=](QVariant newValue) {
        const QString& percent = newValue.toString();
        QRegExp rx("(\\d+)\\%");
        rx.indexIn(percent);
        auto caps = rx.capturedTexts();
        qreal scale = caps[1].toFloat() / 100.;
        QAction act("zoom");
        act.setProperty("ActionType", ZenoMainWindow::ACTION_ZOOM);
        if (m_pEditor)
            m_pEditor->onAction(&act, {scale});
    };
    CallbackCollection cbSet;
    cbSet.cbEditFinished = funcZoomEdited;
    cbZoom = qobject_cast<QComboBox*>(zenoui::createWidget("100%", CONTROL_ENUM, "string", cbSet, props));
    cbZoom->setProperty("focusBorder", "none");
    cbZoom->setEditable(false);
    cbZoom->setFixedSize(ZenoStyle::dpiScaled(60), ZenoStyle::dpiScaled(20));
    cbZoom->view()->setFixedWidth(ZenoStyle::dpiScaled(85));

    pToolLayout->addWidget(pListView);
    pToolLayout->addWidget(pTreeView);

    pToolLayout->addStretch(1);

    pToolLayout->addWidget(pSubnetMgr);
    pToolLayout->addWidget(pFold);
    pToolLayout->addWidget(pUnfold);
    pToolLayout->addWidget(pSnapGrid);
    pToolLayout->addWidget(pShowGrid);
    pToolLayout->addWidget(pCustomParam);
    pToolLayout->addWidget(pGroup);
    pToolLayout->addWidget(pLinkLineShape);

    pToolLayout->addWidget(new ZLineWidget(false, QColor("#121416")));

    pToolLayout->addWidget(m_btnAlways);
    pToolLayout->addWidget(m_btnRun);
    pToolLayout->addWidget(m_btnKill);

    pToolLayout->addStretch(4);

    pToolLayout->addWidget(cbZoom);
    pToolLayout->addWidget(pSearchBtn);
    pToolLayout->addWidget(pSettings);
}

QWidget* DockContent_Editor::initWidget()
{
    ZenoMainWindow* pMainWin = zenoApp->getMainWindow();
    m_pEditor = new ZenoGraphsEditor(pMainWin);
    m_pWidget = m_pEditor;
    m_pEditor->onSubnetListPanel(false, ZenoGraphsEditor::Side_Subnet);     //cihou caofei:
    return m_pEditor;
}

void DockContent_Editor::initConnections()
{
    auto pGraphsMgm = zenoApp->graphsManagment();
    connect(pListView, &ZToolBarButton::toggled, this, [=](bool isShow) 
    { 
        m_pEditor->onSubnetListPanel(isShow, ZenoGraphsEditor::Side_Subnet); 
        pTreeView->setChecked(false);
        pSearchBtn->setChecked(false);
    });
    connect(pTreeView, &ZToolBarButton::toggled, this,[=](bool isShow) 
    { 
        m_pEditor->onSubnetListPanel(isShow, ZenoGraphsEditor::Side_Tree); 
        pSearchBtn->setChecked(false);
        pListView->setChecked(false);
    });
    connect(pSearchBtn, &ZToolBarButton::toggled, this, [=](bool isShow) 
    { 
        m_pEditor->onSubnetListPanel(isShow, ZenoGraphsEditor::Side_Search); 
        pTreeView->setChecked(false);
        pListView->setChecked(false);
    });
    connect(pFold, &ZToolBarButton::clicked, this, [=]() {
        QAction act("Collaspe");
        act.setProperty("ActionType", ZenoMainWindow::ACTION_COLLASPE);
        m_pEditor->onAction(&act);
    });
    connect(pUnfold, &ZToolBarButton::clicked, this, [=]() {
        QAction act("Expand");
        act.setProperty("ActionType", ZenoMainWindow::ACTION_EXPAND);
        m_pEditor->onAction(&act);
    });
    connect(pCustomParam, &ZToolBarButton::clicked, this, [=]() {
        QAction act("CustomUI");
        act.setProperty("ActionType", ZenoMainWindow::ACTION_CUSTOM_UI);
        m_pEditor->onAction(&act);
    });
    connect(pGroup, &ZToolBarButton::clicked, this, [=]() {
        QAction act;
        act.setProperty("ActionType", ZenoMainWindow::ACTION_GROUP);
        m_pEditor->onAction(&act);
    });
    connect(pSnapGrid, &ZToolBarButton::toggled, this, [=](bool bChecked) {
        ZenoSettingsManager::GetInstance().setValue(zsSnapGrid, bChecked);
    });
    connect(pShowGrid, &ZToolBarButton::toggled, this, [=](bool bChecked) {
        ZenoSettingsManager::GetInstance().setValue(zsShowGrid, bChecked);
    });
    connect(pLinkLineShape, &ZToolBarButton::toggled, this, [=](bool bChecked) {
        ZenoSettingsManager::GetInstance().setValue(zsLinkLineShape, bChecked);
    });
    connect(m_pEditor, &ZenoGraphsEditor::zoomed, [=](qreal newFactor) {
        QString percent = QString::number(int(newFactor * 100));
        percent += "%";
        QStringList items;
        QVector<qreal> factors = UiHelper::scaleFactors();
        for (int i = 0; i < factors.size(); i++) {
            qreal factor = factors.at(i);
            if ((i == 0 && factor > newFactor) || (i > 0 && factor > newFactor && factors.at(i - 1) < newFactor)) {
                items.append(percent);
            }
            int per = factor * 100;
            QString sPer = QString("%1%").arg(per);
            if (!items.contains(sPer))
                items.append(sPer);
        }
        cbZoom->clear();
        cbZoom->addItems(items);
        cbZoom->setCurrentText(percent);
    });

    connect(m_btnRun, &ZToolMenuButton::clicked, this, [=]() {
        std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
        ZASSERT_EXIT(mgr);
        ZenoMainWindow *pMainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(pMainWin);
        std::function<void(bool, bool)> setOptixUpdateSeparately = [=](bool updateLightCameraOnly, bool updateMatlOnly) {
            QVector<DisplayWidget *> views = pMainWin->viewports();
            for (auto displayWid : views) {
                if (!displayWid->isGLViewport()) {
                    displayWid->setRenderSeparately(updateLightCameraOnly, updateMatlOnly);
                }
            }
        };
        if (m_btnRun->text() == tr("Run"))
        {
            setOptixUpdateSeparately(false, false);
            mgr->setCacheOpt(ZCacheMgr::Opt_RunAll);
            pMainWin->onRunTriggered();
        }
        else {
            QSettings settings(zsCompanyName, zsEditor);
            if (!settings.value("zencache-enable").toBool()) {
                QMessageBox::warning(nullptr, tr("RunLightCamera"), tr("This function can only be used in cache mode."));
            } else {
                mgr->setCacheOpt(ZCacheMgr::Opt_RunLightCameraMaterial);
                if (m_btnRun->text() == tr("RunLightCamera"))
                {
                    setOptixUpdateSeparately(true, false);
                    pMainWin->onRunTriggered(true, false);
                }
                if (m_btnRun->text() == tr("RunMaterial"))
                {
                    setOptixUpdateSeparately(false, true);
                    pMainWin->onRunTriggered(false, true);
                }
            }
        }
    });

    connect(m_btnKill, &ZToolButton::clicked, this, [=]() {
        killProgram();
    });

    connect(&ZenoSettingsManager::GetInstance(), &ZenoSettingsManager::valueChanged, this, [=](QString name) {
        if (name == zsShowGrid) 
        {
            pShowGrid->setChecked(ZenoSettingsManager::GetInstance().getValue(name).toBool());
        } 
        else if (name == zsSnapGrid) 
        {
            pSnapGrid->setChecked(ZenoSettingsManager::GetInstance().getValue(name).toBool());
        }
    });

    connect(&ZenoSettingsManager::GetInstance(), &ZenoSettingsManager::valueChanged, this, [=](QString key) {
        if (key == ShortCut_Run) {
            m_btnRun->setShortcut(ZenoSettingsManager::GetInstance().getShortCut(ShortCut_Run));
        } else if (key == ShortCut_Kill) {
            m_btnKill->setShortcut(ZenoSettingsManager::GetInstance().getShortCut(ShortCut_Kill));
        }
    });
}

ZenoGraphsEditor* DockContent_Editor::getEditor() const
{
    return m_pEditor;
}

void DockContent_Editor::onCommandDispatched(QAction* pAction, bool bTriggered)
{
    if (m_pEditor)
    {
        m_pEditor->onCommandDispatched(pAction, bTriggered);
    }
}


/// <summary>
/// </summary>
/// <param name="parent"></param>
DockContent_View::DockContent_View(bool bGLView, QWidget* parent)
    : DockToolbarWidget(parent)
    , m_pDisplay(nullptr)
    , m_cbRes(nullptr)
    , m_smooth_shading(nullptr)
    , m_normal_check(nullptr)
    , m_wire_frame(nullptr)
    , m_show_grid(nullptr)
    , m_background_clr(nullptr)
    , m_recordVideo(nullptr)
    , m_screenshoot(nullptr)
    , m_resizeViewport(nullptr)
    , m_bGLView(bGLView)
    , m_background(nullptr)
{
}

void DockContent_View::initToolbar(QHBoxLayout* pToolLayout)
{
    m_moveBtn = new ZToolBarButton(false, ":/icons/viewToolbar_move_idle.svg", ":/icons/viewToolbar_move_light.svg");
    m_moveBtn->setToolTip(tr("Move Object"));

    m_scaleBtn = new ZToolBarButton(false, ":/icons/viewToolbar_scale_idle.svg", ":/icons/viewToolbar_scale_light.svg");
    m_scaleBtn->setToolTip(tr("Scale Object"));

    m_rotateBtn = new ZToolBarButton(false, ":/icons/viewToolbar_rotate_idle.svg", ":/icons/viewToolbar_rotate_light.svg");
    m_rotateBtn->setToolTip(tr("Rotate Object"));

    m_smooth_shading = new ZToolBarButton(true, ":/icons/viewToolbar_smoothshading_idle.svg", ":/icons/viewToolbar_smoothshading_light.svg");
    m_smooth_shading->setToolTip(tr("Smooth Shading"));

    m_normal_check = new ZToolBarButton(true, ":/icons/viewToolbar_normalcheck_idle.svg", ":/icons/viewToolbar_normalcheck_light.svg");
    m_normal_check->setToolTip(tr("Normal Check"));

    m_wire_frame = new ZToolBarButton(true, ":/icons/viewToolbar_wireframe_idle.svg", ":/icons/viewToolbar_wireframe_light.svg");
    m_wire_frame->setToolTip(tr("Wireframe"));

    m_show_grid = new ZToolBarButton(true, ":/icons/viewToolbar_grid_idle.svg", ":/icons/viewToolbar_grid_light.svg");
    m_show_grid->setToolTip(tr("Show Grid"));
    m_show_grid->setChecked(true);

    m_background_clr = new ZToolBarButton(false, ":/icons/viewToolbar_background_idle.svg", ":/icons/viewToolbar_background_light.svg");
    m_background_clr->setToolTip(tr("Background Color"));

    m_recordVideo = new ZToolBarButton(false, ":/icons/viewToolbar_record_idle.svg", ":/icons/viewToolbar_record_light.svg");
    m_recordVideo->setToolTip(tr("Record Video"));


    m_screenshoot = new ZToolBarButton(false, ":/icons/viewToolbar_screenshot_idle.svg", ":/icons/viewToolbar_screenshot_light.svg");
    m_screenshoot->setToolTip(tr("Screenshoot"));

    m_resizeViewport = new ZToolBarButton(false, ":/icons/nodeEditor_fullScreen_unselected.svg", ":/icons/nodeEditor_fullScreen_selected.svg");
    m_resizeViewport->setToolTip(tr("resize viewport"));

    QMenu *pView = new QMenu(tr("View"));
    {
        m_pFocus = new QAction(tr("Focus"));
        //pAction->setShortcut(QKeySequence("F5"));
        QMenu *Viewport = new QMenu(tr("Viewport"));
        m_pOrigin = new QAction(tr("Origin"));
        //pAction->setShortcut(QKeySequence("F5"));
        m_front = new QAction(tr("Front"));
        //m_front->setShortcut(QKeySequence("F5"));
        m_back = new QAction(tr("Back"));
        //m_back->setShortcut(QKeySequence("F5"));
        m_right = new QAction(tr("Right"));
        //m_right->setShortcut(QKeySequence("F5"));
        m_left = new QAction(tr("Left"));
        //m_left->setShortcut(QKeySequence("F5"));
        m_top = new QAction(tr("Top"));
        //m_top->setShortcut(QKeySequence("F5"));
        m_bottom = new QAction(tr("Bottom"));
        //m_bottom->setShortcut(QKeySequence("F5"));

        Viewport->addAction(m_pOrigin);
        Viewport->addAction(m_front);
        Viewport->addAction(m_back);
        Viewport->addAction(m_right);
        Viewport->addAction(m_left);
        Viewport->addAction(m_top);
        Viewport->addAction(m_bottom);

        pView->addAction(m_pFocus);
        pView->addMenu(Viewport);
    }
    QMenu *pObject = new QMenu(tr("Object"));
    {
        QMenu *pTransform = new QMenu(tr("Transform"));
        m_move = new QAction(tr("Move"));

        m_rotate = new QAction(tr("Rotate"));

        m_scale = new QAction(tr("Scale"));

        pTransform->addAction(m_move);
        pTransform->addAction(m_rotate);
        pTransform->addAction(m_scale);

        pObject->addMenu(pTransform);
    }

    QMenuBar *pMenuBar = new QMenuBar(this);
    pMenuBar->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    pMenuBar->setProperty("cssClass", "docktoolbar");
    pMenuBar->setFixedHeight(ZenoStyle::dpiScaled(sToolbarHeight));
    QFont font = QApplication::font();
    font.setWeight(QFont::Medium);
    pMenuBar->setFont(font);

    pMenuBar->addMenu(pView);
    pMenuBar->addMenu(pObject);

    QStringList items = {
        tr("Free"),
        "1024x768",
        "1280x720",
        "1280x768",
        "1280x800",
        "1680x1050",
        "1920x1080",
        tr("Customize Size")
    };
    QVariant props = items;

    QFontMetrics fontMetrics(font);
    Callback_EditFinished funcRender = [=](QVariant newValue) {
        ZASSERT_EXIT(m_pDisplay);
        bool bLock = false;
        if (newValue == tr("Free"))
        {
            nx = 100;
            ny = 100;
        }
        else if (newValue == tr("Customize Size"))
        {
            //todo
            QDialogButtonBox* pButtonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

            QDialog dlg(this);
            QGridLayout* pLayout = new QGridLayout;

            QLineEdit* pWidthEdit = new QLineEdit;
            pWidthEdit->setValidator(new QIntValidator);

            QLineEdit* pHeightEdit = new QLineEdit;
            pHeightEdit->setValidator(new QIntValidator);

            pLayout->addWidget(new QLabel(tr("width")), 0, 0);
            pLayout->addWidget(pWidthEdit, 0, 1);
            pLayout->addWidget(new QLabel(tr("height")), 1, 0);
            pLayout->addWidget(pHeightEdit, 1, 1);
            pLayout->addWidget(pButtonBox, 2, 1);
            dlg.setLayout(pLayout);

            connect(pButtonBox, SIGNAL(accepted()), &dlg, SLOT(accept()));
            connect(pButtonBox, SIGNAL(rejected()), &dlg, SLOT(reject()));

            if (QDialog::Accepted == dlg.exec())
            {
                nx = pWidthEdit->text().toInt();
                ny = pHeightEdit->text().toInt();
                bLock = (nx > 0 && ny > 0);
            }
            else {
                bLock = false;
                nx = 100;
                ny = 100;
            }
        }
        else
        {
            bLock = true;
            QString resStr = newValue.toString();
            auto L = resStr.split('x');
            bool bOK = false;
            nx = L[0].toInt(&bOK);
            ZASSERT_EXIT(nx);
            ny = L[1].toInt(&bOK);
            ZASSERT_EXIT(ny);
        }
        m_pDisplay->setSafeFrames(bLock, nx, ny);
        m_cbRes->setFixedWidth(fontMetrics.horizontalAdvance(newValue.toString()) + ZenoStyle::dpiScaled(28));
    };

    CallbackCollection cbSet;
    cbSet.cbEditFinished = funcRender;
    m_cbRes = qobject_cast<QComboBox*>(zenoui::createWidget("Free", CONTROL_ENUM, "string", cbSet, props));
    m_cbRes->setProperty("focusBorder", "none");
    m_cbRes->setEditable(false);
    m_cbRes->view()->setFixedWidth(ZenoStyle::dpiScaled(110));
    m_cbRes->setFixedSize(fontMetrics.horizontalAdvance(items[0]) + ZenoStyle::dpiScaled(28), ZenoStyle::dpiScaled(20));

    pToolLayout->addWidget(pMenuBar);
    pToolLayout->setAlignment(pMenuBar, Qt::AlignVCenter);
    pToolLayout->addStretch(1);

    pToolLayout->addWidget(m_moveBtn);
    pToolLayout->addWidget(m_rotateBtn);
    pToolLayout->addWidget(m_scaleBtn);

    pToolLayout->addWidget(new ZLineWidget(false, QColor("#121416")));

    pToolLayout->addWidget(m_show_grid);
    pToolLayout->addWidget(m_background_clr);
    pToolLayout->addWidget(m_wire_frame);
    pToolLayout->addWidget(m_smooth_shading);
    pToolLayout->addWidget(m_normal_check);
    {
        m_background = new QCheckBox(tr("Background"));
        m_background->setStyleSheet("color: white;");
        pToolLayout->addWidget(m_background);
    }

    pToolLayout->addWidget(new ZLineWidget(false, QColor("#121416")));
    pToolLayout->addWidget(m_screenshoot);
    pToolLayout->addWidget(m_recordVideo);
    pToolLayout->addWidget(m_resizeViewport);

    pToolLayout->addStretch(7);

    pToolLayout->addWidget(m_cbRes);
}

QWidget* DockContent_View::initWidget()
{
    m_pDisplay = new DisplayWidget(m_bGLView);
    return m_pDisplay;
}

void DockContent_View::initConnections()
{
    connect(m_moveBtn, &ZToolBarButton::clicked, this, [=]() {
        m_pDisplay->changeTransformOperation(0);
    });

    connect(m_rotateBtn, &ZToolBarButton::clicked, this, [=]() {
        m_pDisplay->changeTransformOperation(1);
    });

    connect(m_scaleBtn, &ZToolBarButton::clicked, this, [=]() {
        m_pDisplay->changeTransformOperation(2);
    });

    connect(m_smooth_shading, &ZToolBarButton::toggled, this, [=](bool bToggled) {
        m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_SMOOTH_SHADING, bToggled);
    });

    connect(m_normal_check, &ZToolBarButton::toggled, this, [=](bool bToggled) {
        m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_NORMAL_CHECK, bToggled);
    });

    connect(m_wire_frame, &ZToolBarButton::toggled, this, [=](bool bToggled) {
        m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_WIRE_FRAME, bToggled);
    });

    connect(m_show_grid, &ZToolBarButton::toggled, this, [=](bool bToggled) {
        m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_SHOW_GRID, bToggled);
    });

    connect(m_background_clr, &ZToolBarButton::clicked, this, [=]() {
        m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_BACKGROUND_COLOR, true);
    });

    connect(m_recordVideo, &ZToolBarButton::clicked, this, [=]() {
        m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_RECORD_VIDEO, true);
    });

    connect(m_screenshoot, &ZToolBarButton::clicked, this, [=]() {
        m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_SCREEN_SHOOT, true);
    });
    connect(m_background, &QCheckBox::stateChanged, this, [=](int state) {
        auto &ud = zeno::getSession().userData();
        ud.set2("optix_show_background", state > 0);

        {
            Zenovis *pZenoVis = m_pDisplay->getZenoVis();
            ZASSERT_EXIT(pZenoVis);
            auto session = pZenoVis->getSession();
            ZASSERT_EXIT(session);
            auto scene = session->get_scene();
            ZASSERT_EXIT(scene);
            scene->objectsMan->needUpdateLight = true;
            m_pDisplay->setSimpleRenderOption();
            zenoApp->getMainWindow()->updateViewport();
        }
    });

    connect(m_resizeViewport, &ZToolBarButton::clicked, this, [=]() {

    });
}

void DockContent_View::onCommandDispatched(QAction *pAction, bool bTriggered)
{
    if (m_pDisplay) {
        int actionType = pAction->property("ActionType").toInt();
        m_pDisplay->onCommandDispatched(actionType, bTriggered);
    }
}

DisplayWidget* DockContent_View::getDisplayWid() const
{
    return m_pDisplay;
}

bool DockContent_View::isGLView() const
{
    return m_bGLView;
}

QSize DockContent_View::viewportSize() const
{
    return m_pDisplay->viewportSize();
}

void DockContent_View::onTabAboutToClose()
{
    ZASSERT_EXIT(m_pDisplay);
    if (!m_pDisplay->isGLViewport())
    {
        delete m_pDisplay;
        m_pDisplay = nullptr;
    }
}


int DockContent_View::curResComboBoxIndex()
{
    return m_cbRes->currentIndex();
}

void DockContent_View::setResComboBoxIndex(int index)
{
    QFont fnt = QApplication::font();
    QFontMetrics fontMetrics(fnt);
    m_cbRes->setCurrentIndex(index);
    m_cbRes->setFixedWidth(fontMetrics.horizontalAdvance(m_cbRes->currentText()) + ZenoStyle::dpiScaled(28));
}

std::tuple<int, int> DockContent_View::curResolution()
{
    return std::make_tuple(nx, ny);
}

void DockContent_View::initResolution(int x, int y)
{
    nx = x; ny = y;
}

DockContent_Log::DockContent_Log(QWidget* parent /* = nullptr */)
    : DockToolbarWidget(parent)
    , m_stack(nullptr)
{
}


void DockContent_Log::initToolbar(QHBoxLayout* pToolLayout)
{
    m_pBtnFilterLog = new ZToolBarButton(true, ":/icons/subnet-listview.svg", ":/icons/subnet-listview-on.svg");
    m_pBtnPlainLog = new ZToolBarButton(true, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    m_pDeleteLog = new ZToolBarButton(false, ":/icons/toolbar_delete_idle.svg", ":/icons/toolbar_delete_light.svg");
    m_pBtnPlainLog->setChecked(true);
    m_pBtnFilterLog->setChecked(false);

    pToolLayout->addWidget(m_pBtnPlainLog);
    pToolLayout->addWidget(m_pBtnFilterLog);
    pToolLayout->addStretch();
    pToolLayout->addWidget(m_pDeleteLog);
}

QWidget* DockContent_Log::initWidget()
{
    m_stack = new QStackedWidget;
    m_stack->addWidget(new ZPlainLogPanel);
    m_stack->addWidget(new ZlogPanel);
    m_stack->setCurrentIndex(0);

    m_pWidget = m_stack;
    return m_pWidget;
}

void DockContent_Log::initConnections()
{
    connect(m_pBtnFilterLog, &ZToolBarButton::toggled, this, [=](bool isShow) {
        BlockSignalScope scope(m_pBtnPlainLog);
        m_pBtnPlainLog->setChecked(false);
        m_stack->setCurrentIndex(1);
    });
    connect(m_pBtnPlainLog, &ZToolBarButton::toggled, this, [=](bool isShow) {
        BlockSignalScope scope(m_pBtnFilterLog);
        m_pBtnFilterLog->setChecked(false);
        m_stack->setCurrentIndex(0);
    });
    connect(m_pDeleteLog, &ZToolButton::clicked, this, [=]() {
        zenoApp->logModel()->clear();
        ZPlainLogPanel *pLogger = qobject_cast<ZPlainLogPanel *>(m_stack->widget(0));
        if (pLogger)
            pLogger->clear();
        ZlogPanel* pLogPanel = qobject_cast<ZlogPanel*>(m_stack->widget(1));
        if (pLogPanel)
        {
            pLogPanel->setSyncLog(false);
        }
    });
}


DockContent_Image::DockContent_Image(QWidget *parent)
    : DockToolbarWidget(parent)
    , m_ImagePanel(nullptr)
{
}

QWidget *DockContent_Image::initWidget() {
    m_ImagePanel = new ZenoImagePanel(this);
    return m_ImagePanel;
}

ZenoImagePanel *DockContent_Image::getImagePanel() {
    return m_ImagePanel;
}
