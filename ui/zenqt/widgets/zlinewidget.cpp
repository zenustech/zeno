#include "zlinewidget.h"


ZLineWidget::ZLineWidget(bool bHorizontal, const QColor& clr, QWidget* parent)
	: QFrame(parent)
{
	setFrameShape(bHorizontal ? QFrame::HLine : QFrame::VLine);
	setFrameShadow(QFrame::Plain);
	QPalette pal;
	pal.setColor(QPalette::WindowText, clr);
}

ZPlainLine::ZPlainLine(QWidget* parent)
	: QWidget(parent)
{
    setFixedHeight(1);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setBrush(QPalette::Window, QColor("#000000"));
    setPalette(pal);
}

ZPlainLine::ZPlainLine(int lineWidth, const QColor& clr, QWidget* parent)
	: QWidget(parent)
{
    setFixedHeight(lineWidth);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setAutoFillBackground(true);
    QPalette pal = palette();
	pal.setBrush(QPalette::Window, clr);
    setPalette(pal);
}