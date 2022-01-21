#include "zenographseditor.h"
#include "zenographstabwidget.h"
#include "zenosubnetlistview.h"
#include <comctrl/ztoolbutton.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <model/graphsmodel.h>



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
    m_pSubnetBtn->setBackgroundClr(QColor(36, 36, 36), QColor(36, 36, 36), QColor(36, 36, 36));
    m_pSubnetBtn->setCheckable(true);
    pVLayout->addWidget(m_pSubnetBtn);
    pVLayout->addStretch();
    pVLayout->setSpacing(0);
    pVLayout->setContentsMargins(0, 0, 0, 0);

    pLayout->addLayout(pVLayout);

    QFrame* seperateLine = new QFrame;
    seperateLine->setFrameShape(QFrame::VLine);
    seperateLine->setFrameShadow(QFrame::Plain);
    seperateLine->setLineWidth(2);
    QPalette pal = seperateLine->palette();
    pal.setBrush(QPalette::WindowText, QColor(38, 38, 38));
    seperateLine->setPalette(pal);

    pLayout->addWidget(seperateLine);

    QVBoxLayout* pLayout2 = new QVBoxLayout;

    m_pSubnetList = new ZenoSubnetListPanel();
    m_pSubnetList->hide();
    pLayout->addWidget(m_pSubnetList);

    m_pTabWidget = new ZenoGraphsTabWidget();
    pLayout->addWidget(m_pTabWidget);

    connect(m_pSubnetBtn, SIGNAL(clicked()), this, SLOT(onSubnetBtnClicked()));
    connect(m_pSubnetList, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onListItemActivated(const QModelIndex&)));

    pLayout->setSpacing(0);
    pLayout->setContentsMargins(0, 0, 0, 0);
    setLayout(pLayout);
}

ZenoGraphsEditor::~ZenoGraphsEditor()
{

}

void ZenoGraphsEditor::onListItemActivated(const QModelIndex& index)
{
    const QString& subgraphName = index.data().toString();
    m_pTabWidget->activate(subgraphName);
}

void ZenoGraphsEditor::onModelInited()
{
    GraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    m_pSubnetList->initModel(pModel);
}

void ZenoGraphsEditor::onSubnetBtnClicked()
{
    GraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (pModel == nullptr)
    {
        //open file dialog
    }
    else
    {
        if (m_pSubnetBtn->isChecked())
        {
            m_pSubnetList->show();
        }
        else
        {
            m_pSubnetList->hide();
        }
    }
}