#include "framework.h"
#include "ztabpanel.h"
#include "zpropertiespanel.h"

ZTabPanel::ZTabPanel(QWidget* parent)
    : QTabWidget(parent)
{
    QStackedWidget* propertiesPane = new QStackedWidget;
    propertiesPane->addWidget(new ZPagePropPanel);
    propertiesPane->addWidget(new ZComponentPropPanel);
    propertiesPane->addWidget(new ZElementPropPanel);
    addTab(propertiesPane, "Properties");
    propertiesPane->setCurrentIndex(1);
}

ZTabPanel::~ZTabPanel()
{

}