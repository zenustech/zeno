#ifndef __GRAPHS_MANAGMENT_H__
#define __GRAPHS_MANAGMENT_H__

class GraphsModel;
class GraphsTreeModel;
class ZenoSubGraphScene;

#include <QtWidgets>

class GraphsManagment : public QObject
{
    Q_OBJECT
public:
    GraphsManagment(QObject *parent = nullptr);
    GraphsModel *currentModel();
    GraphsModel *openZsgFile(const QString &fn);
    GraphsModel *importGraph(const QString &fn);
    ZenoSubGraphScene* scene(const QString& subGraphName);
    void reloadGraph(const QString& graphName);
    bool saveCurrent();
    void clear();
    void removeCurrent();
    QList<QAction*> getCategoryActions(QPointF scenePos);

public slots:
    void onNewNodeCreated(const QString& descName, const QPointF& pt);

private:
    GraphsModel *m_model;
    QString m_currFile;
};

#endif