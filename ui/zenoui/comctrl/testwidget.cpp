#include "testwidget.h"
#include <zenoui/comctrl/zwidgetfactory.h>


TestNormalWidget::TestNormalWidget()
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setBrush(QPalette::Background, QColor(49, 54, 72));
    setPalette(pal);

    QString type = "";
    QStringList items = {"23.5 fps", "24 fps", "25 fps", "30 fps", "60 fps"};
    CONTROL_PROPERTIES properties;
    properties["items"] = items;

    CallbackCollection cbSet;

    QWidget* pWidget = zenoui::createWidget("turnLeft", CONTROL_ENUM, type, cbSet, properties);
    pLayout->addWidget(pWidget);

    QWidget* pSlider = zenoui::createWidget(10, CONTROL_HSLIDER, "int", cbSet, properties);
    pLayout->addWidget(pSlider);

    properties["singleStep"] = 1;
    properties["from"] = 1;
    properties["to"] = 100;

    QWidget* pSpinBox = zenoui::createWidget(10, CONTROL_SPINBOX_SLIDER, "int", cbSet, properties);
    pLayout->addWidget(pSpinBox);

    setLayout(pLayout);
}