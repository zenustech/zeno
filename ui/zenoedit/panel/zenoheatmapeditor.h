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
	void setColorRamps(const QLinearGradient& grad);
	QLinearGradient colorRamps() const;
	void updateRampPos(ZenoRampSelector* pSelector);
	void updateRampColor(const QColor& clr);
	void removeRamp();
	void newRamp();
	QGradientStop colorRamp() const;

protected:
	void resizeEvent(QResizeEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;

signals:
	void rampSelected(QGradientStop);

private slots:
	void onSelectionChanged();


private:
	void refreshBar();
	void onResizeInit(QSize sz);

	const int m_szSelector;

	QMap<ZenoRampSelector*, QGradientStop> m_ramps;
	QLinearGradient m_grad;

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
	void mousePressEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void paintEvent(QPaintEvent* event) override;

private:
	void updateColorByMouse(const QPointF& pos);

	QColor m_color;
};

class ZenoHeatMapEditor : public QDialog
{
	Q_OBJECT
public:
	ZenoHeatMapEditor(const QLinearGradient& grad, QWidget* parent = nullptr);
	~ZenoHeatMapEditor();
	QLinearGradient colorRamps() const;

signals:
	void colorPicked(QColor);

protected:
	bool eventFilter(QObject* watched, QEvent* event) override;
	void dragEnterEvent(QDragEnterEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;
	void dropEvent(QDropEvent* event) override;

private slots:
	void onAddRampBtnClicked();
	void onRemoveRampBtnClicked();
	void onRampColorClicked(QGradientStop ramp);
	void setColor(const QColor& clr);
	void onRedChanged(int);
	void onGreenChanged(int);
	void onBlueChanged(int);
	void onHueChanged(int);

	void onCurrentIndexChanged(const QString& text);
	void onClrHexEditFinished();

private:
	void initSignals();
	void init(const QLinearGradient& grad);
	void installFilters();
	void initRamps(const QLinearGradient& grad);
	void initColorView();
	void createDrag(const QPoint& pos, QWidget* widget);

	Ui::HeatMapEditor* m_ui;
};


#endif
