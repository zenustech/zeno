#ifndef __ZENO_HEATMAP_EDITOR_H__
#define __ZENO_HEATMAP_EDITOR_H__

namespace Ui
{
	class HeatMapEditor;
}

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