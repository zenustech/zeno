#ifndef __ZENO_GRAPHS_H__
#define __ZENO_GRAPHS_H__

#include <QtWidgets>

class GraphsModel;
class ZenoSubGraphView;

class ZenoGraphsWidget : public QStackedWidget
{
	Q_OBJECT
public:
    ZenoGraphsWidget(QWidget* parent = nullptr);
    void setGraphsModel(GraphsModel* pModel);
    void initDescriptors();
    QList<QAction*> getCategoryActions(QPointF scenePos);

public slots:
	void onRowsRemoved(const QModelIndex &parent, int first, int last);
    void onSwitchGraph(const QString& graphName);
    void onNewNodeCreated(const QString &descName, const QPointF &pt);

private:
	std::map<QString, ZenoSubGraphView*> m_views;
    GraphsModel* m_model;
    NODE_DESCS m_descs;     //system loaded descs.
};

#endif