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
    GraphsModel* model() const;
    QList<QAction*> getCategoryActions(QPointF scenePos);

public slots:
	void onRowsRemoved(const QModelIndex &parent, int first, int last);
    void onRowsInserted(const QModelIndex &parent, int first, int last);
    void onNewNodeCreated(const QString &descName, const QPointF &pt);

private:
    GraphsModel* m_model;
};

#endif