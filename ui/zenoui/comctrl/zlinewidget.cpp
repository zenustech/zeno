#include "zlinewidget.h"

ZLineWidget::ZLineWidget(bool bHorizontal, const QColor& clr, QWidget* parent)
	: QFrame(parent)
{
	setFrameShape(bHorizontal ? QFrame::HLine : QFrame::VLine);
	setFrameShadow(QFrame::Plain);
	QPalette pal;
	pal.setColor(QPalette::WindowText, clr);
	setPalette(pal);
	setLineWidth(2);
}