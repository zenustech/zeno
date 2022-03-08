#ifndef __ZENO_COLORS_SLIDER_H__
#define __ZENO_COLORS_SLIDER_H__

#include <QtWidgets>

class ZenoColorsSlider : public QSlider
{
	Q_OBJECT
public:
	explicit ZenoColorsSlider(QWidget* parent = nullptr);
	explicit ZenoColorsSlider(Qt::Orientation orientation, QWidget* parent = nullptr);

protected:
	void paintEvent(QPaintEvent* ev) override;
};

#endif