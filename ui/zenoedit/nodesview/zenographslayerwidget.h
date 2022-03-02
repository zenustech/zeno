#ifndef __ZENO_GRAPHS_LAYER_WIDGET_H__
#define __ZENO_GRAPHS_LAYER_WIDGET_H__

#include <QtWidgets>

class ZenoSubGraphView;
class ZIconButton;
class ZenoGraphsLayerWidget;

class LayerPathWidget : public QWidget
{
	Q_OBJECT
public:
	LayerPathWidget(QWidget* parent = nullptr);
	void setPath(const QString& path);
	QString path() const;

private slots:
	void onPathItemClicked();

private:
	QString m_path;
	ZIconButton* m_pForward;
	ZIconButton* m_pBackward;
	ZenoGraphsLayerWidget* m_pLayerWidget;
};

class ZenoStackedViewWidget : public QStackedWidget
{
	Q_OBJECT
public:
	ZenoStackedViewWidget(QWidget* parent = nullptr);
	~ZenoStackedViewWidget();
	void activate(const QString& subGraph, const QString& nodeId = "");
	void clear();

private:
	QMap<QString, ZenoSubGraphView*> m_views;	//rename issues.
};

class ZenoGraphsLayerWidget : public QWidget
{
	Q_OBJECT
public:
	ZenoGraphsLayerWidget(QWidget* parent = nullptr);
	void resetPath(const QString& path, const QString& nodeId);
	void activeByPath(const QString& path);
	void clear();
	QString path() const;

private:
	LayerPathWidget* m_pPathWidget;
	ZenoStackedViewWidget* m_graphsWidget;
};

#endif