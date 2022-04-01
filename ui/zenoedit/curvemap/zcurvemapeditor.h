#ifndef __ZCURVEMAP_EDITOR_H__
#define __ZCURVEMAP_EDITOR_H__

#include <QtWidgets>
#include <zenoui/model/modeldata.h>

class CurveMapView;

class ZCurveMapEditor : public QWidget
{
	Q_OBJECT
public:
	ZCurveMapEditor(QWidget* parent = nullptr);
	~ZCurveMapEditor();
	void init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers);

private:
	CurveMapView* m_view;
};


#endif