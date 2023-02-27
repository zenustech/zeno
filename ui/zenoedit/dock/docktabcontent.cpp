#include "docktabcontent.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zicontoolbutton.h>
#include <zenoui/comctrl/zlabel.h>
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
#include "zenomainwindow.h"


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
    pToolbar->setFixedHeight(ZenoStyle::dpiScaled(28));

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
    m_plblName->setFont(QFont("Segoe UI Bold", 10));
    m_plblName->setMinimumWidth(ZenoStyle::dpiScaled(128));
    QPalette palette = m_plblName->palette();
    palette.setColor(m_plblName->foregroundRole(), QColor("#A3B1C0"));
    m_plblName->setPalette(palette);

    m_pLineEdit = new QLineEdit;
    m_pLineEdit->setText("");
    m_pLineEdit->setProperty("cssClass", "zeno2_2_lineedit");
    m_pLineEdit->setReadOnly(true);
    m_pLineEdit->setFixedHeight(ZenoStyle::dpiScaled(20));

    ZToolBarButton* pFixBtn = new ZToolBarButton(false, ":/icons/fixpanel.svg", ":/icons/fixpanel-on.svg");
    ZToolBarButton* pWikiBtn = new ZToolBarButton(false, ":/icons/wiki.svg", ":/icons/wiki-on.svg");
    m_pSettingBtn = new ZToolBarButton(false, ":/icons/settings.svg", ":/icons/settings-on.svg");

    pToolLayout->addWidget(pIcon);
    pToolLayout->addWidget(m_plblName);
    pToolLayout->addWidget(m_pLineEdit);
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
                m_plblName->setText(idx.data(ROLE_OBJNAME).toString());
                m_pLineEdit->setText(idx.data(ROLE_OBJID).toString());
            }
            else {
                m_plblName->setText("");
                m_pLineEdit->setText("");
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
{
}

void DockContent_Editor::initToolbar(QHBoxLayout* pToolLayout)
{
    pListView = new ZToolBarButton(true, ":/icons/subnet-listview.svg", ":/icons/subnet-listview-on.svg");
    pListView->setChecked(false);

    pTreeView = new ZToolBarButton(true, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    pSubnetMgr = new ZToolBarButton(false, ":/icons/nodeEditor_subnetManager_unselected.svg", ":/icons/nodeEditor_subnetManager_selected.svg");
    pFold = new ZToolBarButton(false, ":/icons/nodeEditor_nodeFold_unselected.svg", ":/icons/nodeEditor_nodeFold_selected.svg");
    pUnfold = new ZToolBarButton(false, ":/icons/nodeEditor_nodeUnfold_unselected.svg", ":/icons/nodeEditor_nodeUnfold_selected.svg");
    pSnapGrid = new ZToolBarButton(true, ":/icons/nodeEditor_snap_unselected.svg", ":/icons/nodeEditor_snap_selected.svg");
    pBlackboard = new ZToolBarButton(false, ":/icons/nodeEditor_blackboard_unselected.svg", ":/icons/nodeEditor_blackboard_selected.svg");
    pFullPanel = new ZToolBarButton(false, ":/icons/nodeEditor_fullScreen_unselected.svg", ":/icons/nodeEditor_fullScreen_selected.svg");
    pSearchBtn = new ZToolBarButton(true, ":/icons/toolbar_search_idle.svg", ":/icons/toolbar_search_light.svg");
    pSettings = new ZToolBarButton(false, ":/icons/toolbar_localSetting_idle.svg", ":/icons/toolbar_localSetting_light.svg");

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
    //cbZoom->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);
    cbZoom->setFixedSize(ZenoStyle::dpiScaled(85), ZenoStyle::dpiScaled(20));

    pToolLayout->addWidget(pListView);
    pToolLayout->addWidget(pTreeView);

    pToolLayout->addSpacing(ZenoStyle::dpiScaled(120));

    pToolLayout->addWidget(pSubnetMgr);
    pToolLayout->addWidget(pFold);
    pToolLayout->addWidget(pUnfold);
    pToolLayout->addWidget(pSnapGrid);
    pToolLayout->addWidget(pBlackboard);
    pToolLayout->addWidget(pFullPanel);
    pToolLayout->addStretch();
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
    connect(pBlackboard, &ZToolBarButton::clicked, this, [=]() {
        QAction act("CustomUI");
        act.setProperty("ActionType", ZenoMainWindow::ACTION_CUSTOM_UI);
        m_pEditor->onAction(&act);
    });
    connect(pSnapGrid, &ZToolBarButton::toggled, this, [=](bool bChecked) {
        QAction act("SnapGrid");
        act.setProperty("ActionType", ZenoMainWindow::ACTION_SNAPGRID);
        if (m_pEditor)
            m_pEditor->onAction(&act, QVariantList(), bChecked);
    });

    connect(m_pEditor, &ZenoGraphsEditor::zoomed, [=](qreal newFactor) {
        QString percent = QString::number(int(newFactor * 100));
        percent += "%";
        cbZoom->setCurrentText(percent);
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
    m_smooth_shading = new ZToolBarButton(true, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    m_smooth_shading->setToolTip(tr("Smooth Shading"));

    m_normal_check = new ZToolBarButton(true, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    m_normal_check->setToolTip(tr("Normal Check"));

    m_wire_frame = new ZToolBarButton(true, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    m_wire_frame->setToolTip(tr("Wireframe"));

    m_show_grid = new ZToolBarButton(true, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    m_show_grid->setToolTip(tr("Show Grid"));
    m_show_grid->setChecked(true);

    m_background_clr = new ZToolBarButton(false, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    m_background_clr->setToolTip(tr("Background Color"));

    m_recordVideo = new ZToolBarButton(false, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    m_recordVideo->setToolTip(tr("Record Video"));

    m_screenshoot = new ZToolBarButton(false, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    m_screenshoot->setToolTip(tr("Screenshoot"));

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

    pToolLayout->addWidget(m_smooth_shading);
    pToolLayout->addWidget(m_normal_check);
    pToolLayout->addWidget(m_wire_frame);
    pToolLayout->addWidget(m_show_grid);
    pToolLayout->addWidget(m_cbRenderWay);
    pToolLayout->addWidget(m_background_clr);
    pToolLayout->addWidget(m_recordVideo);
    pToolLayout->addWidget(m_screenshoot);
    pToolLayout->addStretch();
}

QWidget* DockContent_View::initWidget()
{
    m_pDisplay = new DisplayWidget;
    return m_pDisplay;
}

void DockContent_View::initConnections()
{
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
