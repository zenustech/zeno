#ifndef __ZCURVEMAP_EDITOR_H__
#define __ZCURVEMAP_EDITOR_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/curveutil.h>

using namespace curve_util;

class CurveMapView;
class CurveNodeItem;
class CurveModel;

namespace Ui
{
	class FCurveDlg;
}

class ZCurveMapEditor : public QDialog
{
	Q_OBJECT

public:
    ZCurveMapEditor(bool bTimeline, QWidget* parent = nullptr);
    ~ZCurveMapEditor();
    void addCurve(CurveModel* model);
    void addCurves(const CURVES_DATA& curves);
    int curveCount() const;
    CurveModel *getCurve(int i) const;
    CURVES_MODEL getModel() const;
    CURVES_DATA curves() const;

public slots:
	void onButtonToggled(QAbstractButton* btn, bool bToggled);
	void onNodesSelectionChanged(QList<CurveNodeItem*> lst);
	void onChannelModelDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles = QVector<int>());
    void onNodesDataChanged();
	void onFrameChanged(qreal frame);
	void onLineEditFinished();
    void onLockBtnToggled(bool bToggle);
    void onRangeEdited();
    void onCbTimelineChanged(int);
    void onAddCurveBtnClicked();
    void onDelCurveBtnClicked();

private:
	void init();
	void initUI();
	void initSize();
    void initSignals();
	void initButtonShadow();
    void initStylesheet();
    void initChannelModel();
    CurveModel* currentModel();

	Ui::FCurveDlg* m_ui;

	QButtonGroup* m_pGroupHdlType;
	QMap<QString, CurveModel*> m_models;
	QStandardItemModel* m_channelModel;
	QItemSelectionModel* m_selection;
	bool m_bTimeline;

    std::vector<CurveModel *> m_bate_rows;
};


#endif
