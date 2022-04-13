#ifndef __ZCURVEMAP_EDITOR_H__
#define __ZCURVEMAP_EDITOR_H__

#include <QtWidgets>
#include <zenoui/model/modeldata.h>

class CurveMapView;

namespace Ui
{
	class FCurveDlg;
}

class ZCurveMapEditor : public QDialog
{
	Q_OBJECT
public:
	ZCurveMapEditor(QWidget* parent = nullptr);
	~ZCurveMapEditor();
	void init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers);

private:
	void initUI();

	CurveMapView* m_view;
	Ui::FCurveDlg* m_ui;
};


#endif