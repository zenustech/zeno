#ifndef __ZENO_HEATMAP_EDITOR_H__
#define __ZENO_HEATMAP_EDITOR_H__

namespace Ui
{
	class HeatMapEditor;
}

#include <QtWidgets>
#include <zenoui/model/modeldata.h>

class ZenoRampBar;

class ZenoRampSelector : public QGraphicsEllipseItem
{
	typedef QGraphicsEllipseItem _base;
public:
	ZenoRampSelector(ZenoRampBar* pRampBar, QGraphicsItem* parent = nullptr);
	void initRampPos(const QPointF& pos, const QColor& clr);
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
	static const int m_size = 10;

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

private:
	ZenoRampBar* m_rampBar;
};

class ZenoRampGroove : public QGraphicsLineItem
{
	typedef QGraphicsLineItem _base;
public:
	ZenoRampGroove(QGraphicsItem* parent = nullptr);
};

class ZenoRampBar : public QGraphicsView
{
	Q_OBJECT
public:
	ZenoRampBar(QWidget* parent = nullptr);
	void initRamps(const COLOR_RAMPS& ramps);
	COLOR_RAMPS colorRamps() const;
	void updateRampPos(ZenoRampSelector* pSelector);
	void updateRampColor(const QColor& clr);
	void removeRamp();
	void newRamp();

private:
	void refreshBar();

	const int m_barHeight;
	const int m_szSelector;
	QMap<ZenoRampSelector*, COLOR_RAMP> m_ramps;
	QGraphicsScene* m_scene;
	QGraphicsRectItem* m_pColorItem;
};

class ZenoHeatMapEditor : public QDialog
{
	Q_OBJECT
public:
	ZenoHeatMapEditor(const COLOR_RAMPS& colorRamps, QWidget* parent = nullptr);
	~ZenoHeatMapEditor();
	COLOR_RAMPS colorRamps() const;

signals:
	void colorPicked(QColor);

protected:
	bool eventFilter(QObject* watched, QEvent* event);
	void dragEnterEvent(QDragEnterEvent* event);
	void mousePressEvent(QMouseEvent* event);
	void dropEvent(QDropEvent* event);

private slots:
	void onAddRampBtnClicked();
	void onRemoveRampBtnClicked();

private:
	void initSignals();
	void init();
	void installFilters();
	void initRamps();
	void createDrag(const QPoint& pos, QWidget* widget);

	Ui::HeatMapEditor* m_ui;

	COLOR_RAMPS m_colorRamps;
};


#endif