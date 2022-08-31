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
#include "graphsmanagment.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/comctrl/zlinewidget.h>


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

    ZIconToolButton* pFixBtn = new ZIconToolButton(":/icons/fixpanel.svg", ":/icons/fixpanel-on.svg");
    ZIconToolButton* pWikiBtn = new ZIconToolButton(":/icons/wiki.svg", ":/icons/wiki-on.svg");
    ZIconToolButton* pSettingBtn = new ZIconToolButton(":/icons/settings.svg", ":/icons/settings-on.svg");

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

    ZIconToolButton* pListViewBtn = new ZIconToolButton(":/icons/subnet-listview.svg", ":/icons/subnet-listview.svg");
    pListViewBtn->setCheckable(true);
    ZIconToolButton* pTreeViewBtn = new ZIconToolButton(":/icons/subnet-treeview.svg", ":/icons/subnet-treeview-on.svg");
    pTreeViewBtn->setCheckable(true);
    ZIconToolButton* pSubnetMgr = new ZIconToolButton(":/icons/subnet-mgr.svg", ":/icons/subnet-mgr-on.svg");
    ZIconToolButton* pFold = new ZIconToolButton(":/icons/node-fold.svg", ":/icons/node-fold-on.svg");
    ZIconToolButton* pUnfold = new ZIconToolButton(":/icons/node-unfold.svg", ":/icons/node-unfold-on.svg");
    ZIconToolButton* pSnapGrid = new ZIconToolButton(":/icons/snapgrid.svg", ":/icons/snapgrid-on.svg");
    pSnapGrid->setCheckable(true);
    ZIconToolButton* pBlackboard = new ZIconToolButton(":/icons/blackboard.svg", ":/icons/blackboard-on.svg");
    ZIconToolButton* pFullPanel = new ZIconToolButton(":/icons/full-panel.svg", ":/icons/full-panel-on.svg");

    pToolLayout->addWidget(pListViewBtn);
    pToolLayout->addWidget(pTreeViewBtn);

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
    pVLayout->addWidget(pEditor);
    setLayout(pVLayout);
}