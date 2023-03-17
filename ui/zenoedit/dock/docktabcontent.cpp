#include "docktabcontent.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zicontoolbutton.h>
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zstyleoption.h>
#include "../panel/zenodatapanel.h"
#include "../panel/zenoproppanel.h"
#include "../panel/zenospreadsheet.h"
#include "../panel/zlogpanel.h"
#include "nodesview/zenographseditor.h"
#include "viewport/viewportwidget.h"
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

ZToolRecordingButton::ZToolRecordingButton(const QString &icon, const QString &iconHover, const QString &iconOn,const QString &iconOnHover, const QString &iconPressed)
    : ZToolButton()
{
    setButtonOptions(ZToolButton::Opt_TextLeftToIcon | ZToolButton::Opt_Checkable);
    setIcon(ZenoStyle::dpiScaledSize(QSize(24, 24)), icon, iconHover, iconOn, iconOnHover);
    QFont fnt = zenoApp->font();
    setText(tr("REC"));
    setMargins(ZenoStyle::dpiScaledMargins(QMargins(12, 5, 5, 5)));
    setBackgroundClr(QColor("#383F47"), QColor("#383F47"), QColor("#191D21"), QColor("#191D21"));
    setTextClr(QColor(), QColor(), QColor("#FFFFFF"), QColor("#FFFFFF"));
    m_iconOnPressed = QIcon(iconPressed);
}

void ZToolRecordingButton::paintEvent(QPaintEvent *event) {
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
    option.font = zenoApp->font();
    option.bgRadius = ZenoStyle::dpiScaled(2);
    option.palette.setBrush(QPalette::All, QPalette::Window, QBrush(backgrondColor(option.state)));
    p.drawComplexControl(static_cast<QStyle::ComplexControl>(ZenoStyle::CC_ZenoToolButton), option);
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
    QFont fnt = zenoApp->font();
    m_plblName->setFont(fnt);
    m_plblName->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_plblName->setMinimumWidth(ZenoStyle::dpiScaled(128));
    QPalette palette = m_plblName->palette();
    palette.setColor(m_plblName->foregroundRole(), QColor("#A3B1C0"));
    m_plblName->setPalette(palette);

    ZToolBarButton* pFixBtn = new ZToolBarButton(false, ":/icons/fixpanel.svg", ":/icons/fixpanel-on.svg");
    ZToolBarButton* pWikiBtn = new ZToolBarButton(false, ":/icons/wiki.svg", ":/icons/wiki-on.svg");
    m_pSettingBtn = new ZToolBarButton(false, ":/icons/settings.svg", ":/icons/settings-on.svg");

    pToolLayout->addWidget(pIcon);
    pToolLayout->addWidget(m_plblName);
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
            if (select) {
                m_plblName->setText(idx.data(ROLE_OBJID).toString());
            }
            else {
                m_plblName->setText("");
            }
        }
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
    pFullPanel = new ZToolBarButton(false, ":/icons/nodeEditor_fullScreen_unselected.svg", ":/icons/nodeEditor_fullScreen_selected.svg");
    pSearchBtn = new ZToolBarButton(true, ":/icons/toolbar_search_idle.svg", ":/icons/toolbar_search_light.svg");
    pSettings = new ZToolBarButton(false, ":/icons/toolbar_localSetting_idle.svg", ":/icons/toolbar_localSetting_light.svg");

    m_btnRun = new ZToolButton;
    m_btnKill = new ZToolButton;
    m_btnAlways = new ZToolButton;

    QFont fnt = zenoApp->font();

    m_btnRun->setButtonOptions(ZToolButton::Opt_TextRightToIcon);
    m_btnRun->setIcon(ZenoStyle::dpiScaledSize(QSize(14, 14)), ":/icons/timeline_run_thunder.svg",
                          ":/icons/timeline_run_thunder.svg", "", "");
    m_btnRun->setRadius(ZenoStyle::dpiScaled(2));
    m_btnRun->setFont(fnt);
    m_btnRun->setText(tr("Run"));
    m_btnRun->setCursor(QCursor(Qt::PointingHandCursor));
    m_btnRun->setMargins(ZenoStyle::dpiScaledMargins(QMargins(11, 5, 14, 5)));
    m_btnRun->setBackgroundClr(QColor("#4578AC"), QColor("#4578AC"), QColor("#4578AC"), QColor("#4578AC"));
    m_btnRun->setTextClr(QColor("#FFFFFF"), QColor("#FFFFFF"), QColor("#FFFFFF"), QColor("#FFFFFF"));
    m_btnRun->setShortcut(QKeySequence("F2"));

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
    m_btnKill->setShortcut(QKeySequence("Shift+F2"));

    m_btnAlways->setFixedSize(ZenoStyle::dpiScaledSize(QSize(34, 22)));
    m_btnAlways->setButtonOptions(ZToolButton::Opt_SwitchAnimation);
    m_btnAlways->setIcon(ZenoStyle::dpiScaledSize(QSize(20, 20)), ":/icons/always-off.svg", "", "", "");
    m_btnAlways->setMargins(ZenoStyle::dpiScaledMargins(QMargins(3, 2, 2, 3)));
    m_btnAlways->setBackgroundClr(QColor("#FF191D21"), QColor("#FF191D21"), QColor("#4578AC"), QColor("#4578AC"));
    m_btnAlways->initAnimation();
    if (zeno::envconfig::get("ALWAYS"))
        m_btnAlways->setChecked(true);

    pListView->setChecked(false);
    pShowGrid->setChecked(true);

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
    cbZoom->setEditable(false);
    cbZoom->setFixedSize(ZenoStyle::dpiScaled(85), ZenoStyle::dpiScaled(20));

    pToolLayout->addWidget(pListView);
    pToolLayout->addWidget(pTreeView);

    pToolLayout->addSpacing(ZenoStyle::dpiScaled(120));

    pToolLayout->addWidget(pSubnetMgr);
    pToolLayout->addWidget(pFold);
    pToolLayout->addWidget(pUnfold);
    pToolLayout->addWidget(pSnapGrid);
    pToolLayout->addWidget(pShowGrid);
    pToolLayout->addWidget(pCustomParam);
    pToolLayout->addWidget(pGroup);
    pToolLayout->addWidget(pFullPanel);

    pToolLayout->addStretch();

    pToolLayout->addWidget(m_btnAlways);
    pToolLayout->addWidget(m_btnRun);
    pToolLayout->addWidget(m_btnKill);

    pToolLayout->addSpacing(ZenoStyle::dpiScaled(100));

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

    connect(m_pEditor, &ZenoGraphsEditor::zoomed, [=](qreal newFactor) {
        QString percent = QString::number(int(newFactor * 100));
        percent += "%";
        cbZoom->setCurrentText(percent);
    });

    connect(m_btnRun, &ZToolButton::clicked, this, [=]() {
        ZenoMainWindow* pMainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(pMainWin);
        pMainWin->onRunTriggered();
    });

    connect(m_btnKill, &ZToolButton::clicked, this, [=]() {
        killProgram();
    });

    connect(m_btnAlways, &ZToolButton::toggled, this, [=](bool bChecked) {
        ZenoMainWindow* pMainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(pMainWin);
        if (bChecked)
            pMainWin->onRunTriggered();
        pMainWin->setAlways(bChecked);
    });
    connect(zenoApp->getMainWindow(), &ZenoMainWindow::alwaysModeChanged, this, [=](bool bAlways) {
        m_btnAlways->setChecked(bAlways);
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
DockContent_View::DockContent_View(QWidget* parent)
    : DockToolbarWidget(parent)
    , m_pDisplay(nullptr)
    , m_cbRenderWay(nullptr)
    , m_smooth_shading(nullptr)
    , m_normal_check(nullptr)
    , m_wire_frame(nullptr)
    , m_show_grid(nullptr)
    , m_background_clr(nullptr)
    , m_recordVideo(nullptr)
    , m_screenshoot(nullptr)
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
    QFont font = zenoApp->font();
    font.setWeight(QFont::Medium);
    pMenuBar->setFont(font);

    pMenuBar->addMenu(pView);
    pMenuBar->addMenu(pObject);

    QStringList items = {tr("Solid"), tr("Shading"), tr("Optix")};
    QVariant props = items;

    Callback_EditFinished funcRender = [=](QVariant newValue) {
        if (newValue == items[0]) {
            m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_SOLID, true);
        }
        else if (newValue == items[1]) {
            m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_SHADING, true);
        }
        else if (newValue == items[2]) {
            m_pDisplay->onCommandDispatched(ZenoMainWindow::ACTION_OPTIX, true);
        }
    };
    CallbackCollection cbSet;
    cbSet.cbEditFinished = funcRender;
    m_cbRenderWay = qobject_cast<QComboBox*>(zenoui::createWidget("100%", CONTROL_ENUM, "string", cbSet, props));
    m_cbRenderWay->setEditable(false);
    m_cbRenderWay->setFixedSize(ZenoStyle::dpiScaled(110), ZenoStyle::dpiScaled(20));

    pToolLayout->addWidget(pMenuBar);
    pToolLayout->setAlignment(pMenuBar, Qt::AlignVCenter);
    pToolLayout->addStretch(1);

    pToolLayout->addWidget(m_moveBtn);
    pToolLayout->addWidget(m_rotateBtn);
    pToolLayout->addWidget(m_scaleBtn);

    pToolLayout->addWidget(new ZLineWidget(false, QColor()));

    pToolLayout->addWidget(m_show_grid);
    pToolLayout->addWidget(m_background_clr);
    pToolLayout->addWidget(m_wire_frame);
    pToolLayout->addWidget(m_smooth_shading);
    pToolLayout->addWidget(m_normal_check);

    pToolLayout->addWidget(new ZLineWidget(false, QColor()));
    pToolLayout->addWidget(m_screenshoot);

    pToolLayout->addWidget(m_recordVideo);

    pToolLayout->addStretch(7);

    pToolLayout->addWidget(m_cbRenderWay);
}

QWidget* DockContent_View::initWidget()
{
    m_pDisplay = new DisplayWidget;
    return m_pDisplay;
}

void DockContent_View::initConnections()
{
    connect(m_moveBtn, &ZToolBarButton::clicked, this, [=]() {
        auto viewport = m_pDisplay->getViewportWidget();
        if (viewport)
            viewport->changeTransformOperation(0);
    });

    connect(m_rotateBtn, &ZToolBarButton::clicked, this, [=]() {
        auto viewport = m_pDisplay->getViewportWidget();
        if (viewport)
            viewport->changeTransformOperation(1);
    });

    connect(m_scaleBtn, &ZToolBarButton::clicked, this, [=]() {
        auto viewport = m_pDisplay->getViewportWidget();
        if (viewport)
            viewport->changeTransformOperation(2);
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



DockContent_Log::DockContent_Log(QWidget* parent /* = nullptr */)
    : DockToolbarWidget(parent)
    , m_stack(nullptr)
{
}


void DockContent_Log::initToolbar(QHBoxLayout* pToolLayout)
{
    m_pBtnFilterLog = new ZToolBarButton(true, ":/icons/subnet-listview.svg", ":/icons/subnet-listview-on.svg");
    m_pBtnPlainLog = new ZToolBarButton(true, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    m_pBtnFilterLog->setChecked(true);
    pToolLayout->addWidget(m_pBtnFilterLog);
    pToolLayout->addWidget(m_pBtnPlainLog);
    pToolLayout->addStretch();
}

QWidget* DockContent_Log::initWidget()
{
    m_stack = new QStackedWidget;
    m_stack->addWidget(new ZlogPanel);
    m_stack->addWidget(new ZPlainLogPanel);
    m_stack->setCurrentIndex(0);

    m_pWidget = m_stack;
    return m_pWidget;
}

void DockContent_Log::initConnections()
{
    connect(m_pBtnFilterLog, &ZToolBarButton::toggled, this, [=](bool isShow) {
        BlockSignalScope scope(m_pBtnPlainLog);
        m_pBtnPlainLog->setChecked(false);
        m_stack->setCurrentIndex(0);
    });
    connect(m_pBtnPlainLog, &ZToolBarButton::toggled, this, [=](bool isShow) {
        BlockSignalScope scope(m_pBtnFilterLog);
        m_pBtnFilterLog->setChecked(false);
        m_stack->setCurrentIndex(1);
    });
}
