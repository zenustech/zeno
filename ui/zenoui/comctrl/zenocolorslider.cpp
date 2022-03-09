#include "zenocolorslider.h"


ZenoColorsSlider::ZenoColorsSlider(QWidget* parent)
	: QSlider(parent)
{

}

ZenoColorsSlider::ZenoColorsSlider(Qt::Orientation orientation, QWidget* parent)
	: QSlider(orientation, parent)
{

}

void ZenoColorsSlider::paintEvent(QPaintEvent* ev)
{
	//Q_D(QSlider);
	QPainter p(this);
	QStyleOptionSlider opt;
	initStyleOption(&opt);

	opt.subControls = QStyle::SC_SliderHandle;
	if (tickPosition() != NoTicks)
		opt.subControls |= QStyle::SC_SliderTickmarks;
	//if (d->pressedControl) {
	//	opt.activeSubControls = d->pressedControl;
	//	opt.state |= QStyle::State_Sunken;
	//}
	//else {
	//	opt.activeSubControls = d->hoverControl;
	//}

	style()->drawComplexControl(QStyle::CC_Slider, &opt, &p, this);

	//QSlider::paintEvent(ev);
}