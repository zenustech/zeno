#include "framework.h"
#include "zpropertiespanel.h"


ValueInputWidget::ValueInputWidget(const QString& name, QWidget* parent)
    : QWidget(parent)
    , m_pSpinBox(nullptr)
    , m_pLineEdit(nullptr)
{
    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->addWidget(new QLabel(name));

    //m_pSpinBox = new QSpinBox;
    //pLayout->addWidget(m_pSpinBox);

    m_pLineEdit = new QLineEdit;
    pLayout->addWidget(m_pLineEdit);

   

    setLayout(pLayout);
}


ZPagePropPanel::ZPagePropPanel(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pVBoxLayout = new QVBoxLayout;

    pVBoxLayout->addWidget(new QLabel("Grid Settings"));

    QLabel* pLabel1 = new QLabel("Transform");
    pVBoxLayout->addWidget(pLabel1);

    QHBoxLayout* pHLayout = new QHBoxLayout;
    m_pWidth = new ValueInputWidget("W:");
    m_pHeight = new ValueInputWidget("H:");
    pHLayout->addWidget(m_pWidth);
    pHLayout->addWidget(m_pHeight);
    pVBoxLayout->addLayout(pHLayout);

    QFrame* pLine = new QFrame;
    pLine->setFrameShape(QFrame::HLine);
    pVBoxLayout->addWidget(pLine);

    pVBoxLayout->addWidget(new QLabel("Color"));
    pVBoxLayout->addStretch();

    setLayout(pVBoxLayout);
}

ZComponentPropPanel::ZComponentPropPanel(QWidget* parent)
    : QWidget(parent)
    , m_pX(new ValueInputWidget("X:"))
    , m_pY(new ValueInputWidget("Y:"))
    , m_pWidth(new ValueInputWidget("W:"))
    , m_pHeight(new ValueInputWidget("H:"))
{
    QVBoxLayout* pVBoxLayout = new QVBoxLayout;

    pVBoxLayout->addWidget(new QLabel("Transform"));

    QGridLayout* pLayout = new QGridLayout;

    m_pX = new ValueInputWidget("X:");
    m_pY = new ValueInputWidget("Y:");
    m_pWidth = new ValueInputWidget("W:");
    m_pHeight = new ValueInputWidget("H:");

    pLayout->addWidget(m_pX, 0, 0);
    pLayout->addWidget(m_pWidth, 0, 1);
    pLayout->addWidget(m_pY, 1, 0);
    pLayout->addWidget(m_pHeight, 1, 1);

    pVBoxLayout->addLayout(pLayout);
    pVBoxLayout->addStretch();

    setLayout(pVBoxLayout);
}

ZElementPropPanel::ZElementPropPanel(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pVBoxLayout = new QVBoxLayout;
    pVBoxLayout->addWidget(new QLabel("assets"));
    
    QHBoxLayout* pHLayout = new QHBoxLayout;
    m_pAsset = new QLabel("example.svg");
    pHLayout->addWidget(m_pAsset);

    QPushButton* pBtnFile = new QPushButton("...");
    pHLayout->addWidget(pBtnFile);

    pVBoxLayout->addLayout(pHLayout);

    QFrame* pLine = new QFrame;
    pLine->setFrameShape(QFrame::HLine);
    pVBoxLayout->addWidget(pLine);

    pVBoxLayout->addWidget(new QLabel("Transform"));

    QGridLayout* pLayout = new QGridLayout;

    m_pX = new ValueInputWidget("X:");
    m_pY = new ValueInputWidget("Y:");
    m_pWidth = new ValueInputWidget("W:");
    m_pHeight = new ValueInputWidget("H:");

    pLayout->addWidget(m_pX, 0, 0);
    pLayout->addWidget(m_pWidth, 0, 1);
    pLayout->addWidget(m_pY, 1, 0);
    pLayout->addWidget(m_pHeight, 1, 1);

    pVBoxLayout->addLayout(pLayout);

    pVBoxLayout->addStretch();
    
    setLayout(pVBoxLayout);
}