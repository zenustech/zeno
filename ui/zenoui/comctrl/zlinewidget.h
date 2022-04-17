#ifndef __ZLINE_WIDGET_H__
#define __ZLINE_WIDGET_H__

#include <QFrame>

class ZLineWidget : public QFrame
{
	Q_OBJECT
public:
	ZLineWidget(bool bHorizontal, const QColor& clr, QWidget* parent = nullptr);
};


#endif