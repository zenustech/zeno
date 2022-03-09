#ifndef __ZENO_HEATMAP_EDITOR_H__
#define __ZENO_HEATMAP_EDITOR_H__

namespace Ui
{
	class HeatMapEditor;
}

#include <QtWidgets>

class ZenoRampSelector : public QGraphicsEllipseItem
{
	typedef QGraphicsEllipseItem _base;
public:
	ZenoRampSelector(const QColor& clr, int y, QGraphicsItem* parent = nullptr);
	void setColor(const QColor& clr);
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

private:
	QColor m_color;
	int m_y;
	static const int m_size = 12;
};

class ZenoRampBar : public QGraphicsView
{
	Q_OBJECT
public:
	ZenoRampBar(QWidget* parent = nullptr);

private:
	QGraphicsScene* m_scene;
};

class ZenoHeatMapEditor : public QWidget
{
	Q_OBJECT
public:
	ZenoHeatMapEditor(QWidget* parent = nullptr);

signals:
	void colorPicked(QColor);

protected:
	bool eventFilter(QObject* watched, QEvent* event);
	void dragEnterEvent(QDragEnterEvent* event);
	void mousePressEvent(QMouseEvent* event);
	void dropEvent(QDropEvent* event);

private:
	void initSignals();
	void init();
	void installFilters();
	void initRamps();
	void createDrag(const QPoint& pos, QWidget* widget);

	Ui::HeatMapEditor* m_ui;
};


#endif