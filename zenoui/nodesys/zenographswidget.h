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

public slots:
	void onRowsRemoved(const QModelIndex &parent, int first, int last);
    void onSwitchGraph(const QString& graphName);

private:
	std::map<QString, ZenoSubGraphView*> m_views;
    GraphsModel* m_model;
};

#endif