#include "zenographseditor.h"
#include "zenographstabwidget.h"
#include "zenosubnetlistview.h"
#include <comctrl/ztoolbutton.h>


ZenoGraphsEditor::ZenoGraphsEditor(QWidget* parent)
    : QWidget(parent)
    , m_pSubnetBtn(nullptr)
    , m_pSubnetList(nullptr)
    , m_pTabWidget(nullptr)
{
    QHBoxLayout* pLayout = new QHBoxLayout;

    QVBoxLayout* pVLayout = new QVBoxLayout;
    m_pSubnetBtn = new ZToolButton(
        ZToolButton::Opt_HasIcon | ZToolButton::Opt_HasText | ZToolButton::Opt_UpRight,
        QIcon(":/icons/subnetbtn.svg"),
        QSize(20, 20),
        "Subset",
        nullptr
    );
    pVLayout->addWidget(m_pSubnetBtn);
    pVLayout->addStretch();

    pLayout->setSpacing(0);
    pLayout->addLayout(pVLayout);

    m_pSubnetList = new ZenoSubnetListView();
    pLayout->addWidget(m_pSubnetList);

    m_pTabWidget = new ZenoGraphsTabWidget();
    pLayout->addWidget(m_pTabWidget);

    setLayout(pLayout);
}

ZenoGraphsEditor::~ZenoGraphsEditor()
{

}