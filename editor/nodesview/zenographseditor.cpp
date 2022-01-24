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
    , m_seperateLine(nullptr)
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
    m_pSubnetBtn->hide();
    pVLayout->addWidget(m_pSubnetBtn);
    pVLayout->addStretch();
    pVLayout->setSpacing(0);
    pVLayout->setContentsMargins(0, 0, 0, 0);

    pLayout->addLayout(pVLayout);

    m_seperateLine = new QFrame;
    m_seperateLine->setFrameShape(QFrame::VLine);
    m_seperateLine->setFrameShadow(QFrame::Plain);
    m_seperateLine->setLineWidth(2);
    QPalette pal = m_seperateLine->palette();
    pal.setBrush(QPalette::WindowText, QColor(38, 38, 38));
    m_seperateLine->setPalette(pal);
    m_seperateLine->hide();

    pLayout->addWidget(m_seperateLine);

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

void ZenoGraphsEditor::resetModel(GraphsModel* pModel)
{
    m_pSubnetList->initModel(pModel);
    m_pTabWidget->resetModel(pModel);
    m_pSubnetBtn->show();
    m_pSubnetBtn->setChecked(true);
    m_pSubnetList->show();
    m_seperateLine->show();
    connect(pModel, SIGNAL(modelReset()), this, SLOT(onCurrentModelClear()));
}

void ZenoGraphsEditor::onCurrentModelClear()
{
    m_pSubnetList->hide();
    m_pSubnetBtn->hide();
    m_seperateLine->hide();
    m_pTabWidget->clear();
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
            m_seperateLine->show();
        }
        else
        {
            m_pSubnetList->hide();
            m_seperateLine->hide();
        }
    }
}