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
	void initRamps(int width);
	void setColorRamps(const COLOR_RAMPS& ramps);
	COLOR_RAMPS colorRamps() const;
	void updateRampPos(ZenoRampSelector* pSelector);
	void updateRampColor(const QColor& clr);
	void removeRamp();
	void newRamp();
	COLOR_RAMP colorRamp() const;

protected:
	void resizeEvent(QResizeEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;

signals:
	void rampSelected(COLOR_RAMP);

private slots:
	void onSelectionChanged();

private:
	void refreshBar();

	const int m_barHeight;
	const int m_szSelector;
	int m_width;

	COLOR_RAMPS m_initRamps;
	QMap<ZenoRampSelector*, COLOR_RAMP> m_ramps;

	ZenoRampSelector* m_currSelector;
	QGraphicsScene* m_scene;
	QGraphicsRectItem* m_pColorItem;
	ZenoRampGroove* m_pLineItem;
};

class SVColorView : public QWidget
{
	Q_OBJECT
public:
	SVColorView(QWidget* parent = nullptr);
	QColor color() const;

signals:
	void colorChanged(const QColor& clr);

public slots:
	void setColor(const QColor& clr);

protected:
	void mousePressEvent(QMouseEvent* event);
	void mouseMoveEvent(QMouseEvent* event);
	void paintEvent(QPaintEvent* event);

private:
	void updateColorByMouse(const QPointF& pos);

	QColor m_color;
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
	void onRampColorClicked(COLOR_RAMP ramp);
	void setColor(const QColor& clr);
	void onRedChanged(int);
	void onGreenChanged(int);
	void onBlueChanged(int);
	void onHueChanged(int);

private:
	void initSignals();
	void init();
	void installFilters();
	void initRamps();
	void initColorView();
	void createDrag(const QPoint& pos, QWidget* widget);

	Ui::HeatMapEditor* m_ui;

	COLOR_RAMPS m_colorRamps;
};


#endif