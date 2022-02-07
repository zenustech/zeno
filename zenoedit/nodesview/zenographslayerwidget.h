#ifndef __ZENO_GRAPHS_LAYER_WIDGET_H__
#define __ZENO_GRAPHS_LAYER_WIDGET_H__

#include <QtWidgets>

class LayerPathWidget : public QWidget
{
	Q_OBJECT
public:
	LayerPathWidget(QWidget* parent = nullptr);
	void setPath(const QString& path);

private:
	QString m_path;
};

class ZenoGraphsLayerWidget : public QWidget
{
	Q_OBJECT
public:
	ZenoGraphsLayerWidget(QWidget* parent = nullptr);
	void resetPath(const QString& path);

private:
	LayerPathWidget* m_pPathWidget;
	QStackedWidget* m_graphsWidget;
};

#endif