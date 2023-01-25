#include "docktabcontent.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zicontoolbutton.h>
#include <zenoui/comctrl/zlabel.h>
#include "../panel/zenodatapanel.h"
#include "../panel/zenoproppanel.h"
#include "../panel/zenospreadsheet.h"
#include "../panel/zlogpanel.h"
#include "nodesview/zenographseditor.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/modelrole.h>
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


DockContent_Parameter::DockContent_Parameter(QWidget* parent)
    : QWidget(parent)
{
    QHBoxLayout* pToolLayout = new QHBoxLayout;
    pToolLayout->setContentsMargins(ZenoStyle::dpiScaled(8), ZenoStyle::dpiScaled(4),
        ZenoStyle::dpiScaled(4), ZenoStyle::dpiScaled(4));

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
    ZToolBarButton* pSettingBtn = new ZToolBarButton(false, ":/icons/settings.svg", ":/icons/settings-on.svg");

    pToolLayout->addWidget(pIcon);
    pToolLayout->addWidget(m_plblName);
    pToolLayout->addWidget(m_pLineEdit);
    pToolLayout->addStretch();
    pToolLayout->addWidget(pFixBtn);
    pToolLayout->addWidget(pWikiBtn);
    pToolLayout->addWidget(pSettingBtn);
    pToolLayout->setSpacing(9);

    QVBoxLayout* pVLayout = new QVBoxLayout;
    pVLayout->addLayout(pToolLayout);
    pVLayout->setContentsMargins(0, 0, 0, 0);
    pVLayout->setSpacing(0);

    ZenoPropPanel* prop = new ZenoPropPanel;
    pVLayout->addWidget(prop);
    setLayout(pVLayout);

    connect(pSettingBtn, &ZToolBarButton::clicked, prop, &ZenoPropPanel::onSettings);
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
    : QWidget(parent)
    , m_pEditor(nullptr)
{
    QHBoxLayout* pToolLayout = new QHBoxLayout;
    pToolLayout->setContentsMargins(ZenoStyle::dpiScaled(8), ZenoStyle::dpiScaled(4),
        ZenoStyle::dpiScaled(4), ZenoStyle::dpiScaled(4));
    pToolLayout->setSpacing(ZenoStyle::dpiScaled(5));

    ZenoMainWindow* pMainWin = zenoApp->getMainWindow();
    m_pEditor = new ZenoGraphsEditor(pMainWin);

    ZToolBarButton* pListView = new ZToolBarButton(true, ":/icons/subnet-listview.svg", ":/icons/subnet-listview-on.svg");
    pListView->setChecked(true);

    ZToolBarButton* pTreeView = new ZToolBarButton(true, ":/icons/nodeEditor_nodeTree_unselected.svg", ":/icons/nodeEditor_nodeTree_selected.svg");
    ZToolBarButton* pSubnetMgr = new ZToolBarButton(false, ":/icons/nodeEditor_subnetManager_unselected.svg", ":/icons/nodeEditor_subnetManager_selected.svg");
    ZToolBarButton* pFold = new ZToolBarButton(false, ":/icons/nodeEditor_nodeFold_unselected.svg", ":/icons/nodeEditor_nodeFold_selected.svg");
    ZToolBarButton* pUnfold = new ZToolBarButton(false, ":/icons/nodeEditor_nodeUnfold_unselected.svg", ":/icons/nodeEditor_nodeUnfold_selected.svg");
    ZToolBarButton* pSnapGrid = new ZToolBarButton(true, ":/icons/nodeEditor_snap_unselected.svg", ":/icons/nodeEditor_snap_selected.svg");
    ZToolBarButton* pBlackboard = new ZToolBarButton(false, ":/icons/nodeEditor_blackboard_unselected.svg", ":/icons/nodeEditor_blackboard_selected.svg");
    ZToolBarButton* pFullPanel = new ZToolBarButton(false, ":/icons/nodeEditor_fullScreen_unselected.svg", ":/icons/nodeEditor_fullScreen_selected.svg");
    ZToolBarButton* pSearchBtn = new ZToolBarButton(true, ":/icons/toolbar_search_idle.svg", ":/icons/toolbar_search_light.svg");
    ZToolBarButton* pSettings = new ZToolBarButton(false, ":/icons/toolbar_localSetting_idle.svg", ":/icons/toolbar_localSetting_light.svg");

    QStringList items = {"25%", "50%", "75%", "100%", "125%", "150%", "200%", "300%", "400%", "500%"};
    QVariant props = items;

    Callback_EditFinished funcZoomEdited = [=](QVariant newValue) {
        const QString& percent = newValue.toString();
        QRegExp rx("(\\d+)\\%");
        rx.indexIn(percent);
        auto caps = rx.capturedTexts();
        qreal scale = caps[1].toFloat() / 100.;
        QAction act("zoom");
        act.setProperty("ActionType", ZenoMainWindow::ACTION_ZOOM);
        m_pEditor->onAction(&act, {scale});
    };
    CallbackCollection cbSet;
    cbSet.cbEditFinished = funcZoomEdited;
    QComboBox* cbZoom = qobject_cast<QComboBox*>(zenoui::createWidget("100%", CONTROL_ENUM, "string", cbSet, props));
    cbZoom->setEditable(false);
    connect(m_pEditor, &ZenoGraphsEditor::zoomed, [=](qreal newFactor) {
        QString percent = QString::number(int(newFactor * 100));
        percent += "%";
        cbZoom->setCurrentText(percent);
    });

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

    QVBoxLayout* pVLayout = new QVBoxLayout;
    pVLayout->setContentsMargins(0, 0, 0, 0);
    pVLayout->setSpacing(0);

    pVLayout->addLayout(pToolLayout);

    //add the seperator line
    ZPlainLine* pLine = new ZPlainLine(1, QColor("#000000"));
    pVLayout->addWidget(pLine);

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

    pVLayout->addWidget(m_pEditor);
    setLayout(pVLayout);
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


DockContent_View::DockContent_View(QWidget* parent)
    : QWidget(parent)
{

}


DockContent_Log::DockContent_Log(QWidget* parent /* = nullptr */)
    : QWidget(parent)
{
    QHBoxLayout* pToolLayout = new QHBoxLayout;
    pToolLayout->setContentsMargins(ZenoStyle::dpiScaled(8), ZenoStyle::dpiScaled(4),
        ZenoStyle::dpiScaled(4), ZenoStyle::dpiScaled(4));
    pToolLayout->setSpacing(ZenoStyle::dpiScaled(5));

    QVBoxLayout* pVLayout = new QVBoxLayout;
    pVLayout->setContentsMargins(0, 0, 0, 0);
    pVLayout->setSpacing(0);

    pVLayout->addLayout(pToolLayout);
    //add the seperator line
    pVLayout->addWidget(new ZPlainLine(1, QColor("#000000")));

    ZlogPanel* logPanel = new ZlogPanel;
    pVLayout->addWidget(logPanel);
    setLayout(pVLayout);
}

