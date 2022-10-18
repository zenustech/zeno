#include "testwidget.h"
#include <zenoui/comctrl/zwidgetfactory.h>


TestNormalWidget::TestNormalWidget()
{
    QVBoxLayout* pLayout = new QVBoxLayout;

    QString type = "";
    QStringList items = {"23.5 fps", "24 fps", "25 fps", "30 fps", "60 fps"};
    QWidget* pWidget = zenoui::createWidget("turnLeft", CONTROL_ENUM, type, [=](QVariant status) {
        int j;
        j = 0;
    }, [=](bool) {}, items);
    pLayout->addWidget(pWidget);

    setLayout(pLayout);
}