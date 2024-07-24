#include "testwidget.h"
#include "widgets/zwidgetfactory.h"
#include "uicommon.h"
#include "reflect/reflection.generated.hpp"


TestNormalWidget::TestNormalWidget()
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setBrush(QPalette::Background, QColor(49, 54, 72));
    setPalette(pal);

    QString type = "";
    std::vector<std::string> items = {"23.5 fps", "24 fps", "25 fps", "30 fps", "60 fps"};
    zeno::reflect::Any properties = items;

    CallbackCollection cbSet;

    QWidget* pWidget = zenoui::createWidget("turnLeft", zeno::Combobox, zeno::Param_Null, cbSet, properties);
    pLayout->addWidget(pWidget);

    std::vector<float> sliderInfo = { 1.0, 100.0, 1.0 };
    properties = sliderInfo;

    QWidget* pSlider = zenoui::createWidget(10, zeno::Slider, zeno::Param_Int, cbSet, properties);
    pLayout->addWidget(pSlider);

    QWidget* pSpinBox = zenoui::createWidget(10, zeno::SpinBoxSlider, zeno::Param_Int, cbSet, properties);
    pLayout->addWidget(pSpinBox);

    setLayout(pLayout);
}