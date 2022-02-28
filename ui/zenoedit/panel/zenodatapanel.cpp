#include "zenodatapanel.h"
#include "zenodatamodel.h"


ZenoDataTable::ZenoDataTable(QWidget *parent)
    : QTableView(parent)
    , m_model(nullptr)
{
    init();
}

ZenoDataTable::~ZenoDataTable()
{
}

void ZenoDataTable::init()
{
    m_model = new NodeDataModel;
    setModel(m_model);
}


ZenoDataPanel::ZenoDataPanel(QWidget *parent)
    : QWidget(parent)
{
    init();
}

ZenoDataPanel::~ZenoDataPanel()
{
}

void ZenoDataPanel::init()
{
    QVBoxLayout *pVLayout = new QVBoxLayout;
    ZenoDataTable *pDataTable = new ZenoDataTable;
    pVLayout->addWidget(pDataTable);
    setLayout(pVLayout);
}