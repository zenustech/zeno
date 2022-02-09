#ifndef __ZENO_GRAPHS_LAYER_WIDGET_H__
#define __ZENO_GRAPHS_LAYER_WIDGET_H__

#include <QtWidgets>

class ZenoSubGraphView;
class ZIconButton;

class LayerPathWidget : public QWidget
{
	Q_OBJECT
public:
	LayerPathWidget(QWidget* parent = nullptr);
	void setPath(const QString& path);

private:
	QString m_path;
	ZIconButton* m_pForward;
	ZIconButton* m_pBackward;
};

class ZenoStackedViewWidget : public QStackedWidget
{
	Q_OBJECT
public:
	ZenoStackedViewWidget(QWidget* parent = nullptr);
	~ZenoStackedViewWidget();
	void activate(const QString& subGraph, const QString& nodeId = "");

private:
	QMap<QString, ZenoSubGraphView*> m_views;
};

class ZenoGraphsLayerWidget : public QWidget
{
	Q_OBJECT
public:
	ZenoGraphsLayerWidget(QWidget* parent = nullptr);
	void resetPath(const QString& path, const QString& nodeId);

private:
	LayerPathWidget* m_pPathWidget;
	ZenoStackedViewWidget* m_graphsWidget;
};

#endif