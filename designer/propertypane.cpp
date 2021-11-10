#include "framework.h"
#include "propertypane.h"

PropertyPane::PropertyPane(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pLayout = new QVBoxLayout;

    pLayout->addWidget(new QLabel(tr("Inspector")));
    pLayout->addStretch();
    
    setLayout(pLayout);
}