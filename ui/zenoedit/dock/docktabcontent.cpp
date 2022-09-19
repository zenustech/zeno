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


ZToolBarButton::ZToolBarButton(bool bCheckable, const QString& icon, const QString& iconOn)
    : ZToolButton(ZToolButton::Opt_HasIcon, icon, iconOn)
{
    setCheckable(bCheckable);

    QColor bgOn("#4F5963");

    int marginLeft = ZenoStyle::dpiScaled(5);
    int marginRight = ZenoStyle::dpiScaled(5);
    int marginTop = ZenoStyle::dpiScaled(2);
    int marginBottom = ZenoStyle::dpiScaled(2);

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
{
    QHBoxLayout* pToolLayout = new QHBoxLayout;
    pToolLayout->setContentsMargins(ZenoStyle::dpiScaled(8), ZenoStyle::dpiScaled(4),
        ZenoStyle::dpiScaled(4), ZenoStyle::dpiScaled(4));
    pToolLayout->setSpacing(ZenoStyle::dpiScaled(5));

    ZToolBarButton* pListView = new ZToolBarButton(true, ":/icons/subnet-listview.svg", ":/icons/subnet-listview-on.svg");
    pListView->setChecked(true);

    ZToolBarButton* pTreeView = new ZToolBarButton(true, ":/icons/subnet-treeview.svg", ":/icons/subnet-treeview-on.svg");
    ZToolBarButton* pSubnetMgr = new ZToolBarButton(false, ":/icons/subnet-mgr.svg", ":/icons/subnet-mgr-on.svg");
    ZToolBarButton* pFold = new ZToolBarButton(false, ":/icons/node-fold.svg", ":/icons/node-fold-on.svg");
    ZToolBarButton* pUnfold = new ZToolBarButton(false, ":/icons/node-unfold.svg", ":/icons/node-unfold-on.svg");
    ZToolBarButton* pSnapGrid = new ZToolBarButton(true, ":/icons/snapgrid.svg", ":/icons/snapgrid-on.svg");
    ZToolBarButton* pBlackboard = new ZToolBarButton(false, ":/icons/blackboard.svg", ":/icons/blackboard-on.svg");
    ZToolBarButton* pFullPanel = new ZToolBarButton(false, ":/icons/full-panel.svg", ":/icons/full-panel-on.svg");

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

    QVBoxLayout* pVLayout = new QVBoxLayout;
    pVLayout->setContentsMargins(0, 0, 0, 0);
    pVLayout->setSpacing(0);

    pVLayout->addLayout(pToolLayout);

    //add the seperator line
    ZPlainLine* pLine = new ZPlainLine(1, QColor("#000000"));
    pVLayout->addWidget(pLine);

    ZenoMainWindow* pMainWin = zenoApp->getMainWindow();
    ZenoGraphsEditor* pEditor = new ZenoGraphsEditor(pMainWin);

    connect(pListView, &ZToolBarButton::toggled, pEditor, &ZenoGraphsEditor::onSubnetListPanel);
    connect(pFold, &ZToolBarButton::clicked, this, [=]() {
        pEditor->onAction(pEditor->tr("Collaspe"));
    });
    connect(pUnfold, &ZToolBarButton::clicked, this, [=]() {
        pEditor->onAction(pEditor->tr("Expand"));
    });

    pVLayout->addWidget(pEditor);
    setLayout(pVLayout);
}

DockContent_View::DockContent_View(QWidget* parent)
    : QWidget(parent)
{

}