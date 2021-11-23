#include "framework.h"
#include "ztabpanel.h"
#include "zpropertiespanel.h"

ZTabPanel::ZTabPanel(QWidget* parent)
    : QTabWidget(parent)
    , m_pagePanel(new ZPagePropPanel)
    , m_componentPanel(new ZComponentPropPanel)
    , m_elementPanel(new ZElementPropPanel)
{
    QStackedWidget* propertiesPane = new QStackedWidget;
    propertiesPane->addWidget(m_pagePanel);
    propertiesPane->addWidget(m_componentPanel);
    propertiesPane->addWidget(m_elementPanel);
    addTab(propertiesPane, "Properties");
    propertiesPane->setCurrentIndex(1);
}

ZTabPanel::~ZTabPanel()
{

}

void ZTabPanel::resetModel()
{
    m_componentPanel->initModel();
}