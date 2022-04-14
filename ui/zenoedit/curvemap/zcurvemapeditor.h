#ifndef __ZCURVEMAP_EDITOR_H__
#define __ZCURVEMAP_EDITOR_H__

#include <QtWidgets>
#include <zenoui/model/modeldata.h>
#include "curveutil.h"

using namespace curve_util;

class CurveMapView;
class CurveNodeItem;

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

public slots:
	void onButtonToggled(QAbstractButton* btn, bool bToggled);
	void onNodesSelectionChanged(QList<CurveNodeItem*> lst);
    void onNodesDataChanged();
	void onLineEditFinished();

private:
	void initUI();
	void initSize();
    void initSignals();
	void initButtonShadow();
    void initStylesheet();

	Ui::FCurveDlg* m_ui;

	QButtonGroup* m_pGroupHdlType;

};


#endif