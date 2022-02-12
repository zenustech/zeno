#ifndef __GRAPHS_MANAGMENT_H__
#define __GRAPHS_MANAGMENT_H__

class GraphsModel;
class GraphsTreeModel;
class ZenoSubGraphScene;

#include <QtWidgets>

#include <zenoui/include/igraphsmodel.h>

class GraphsManagment : public QObject
{
    Q_OBJECT
public:
    GraphsManagment(QObject *parent = nullptr);
    IGraphsModel* currentModel();
    IGraphsModel* openZsgFile(const QString &fn);
    IGraphsModel* importGraph(const QString &fn);
    ZenoSubGraphScene* scene(const QString& subGraphName);
    void reloadGraph(const QString& graphName);
    bool saveCurrent();
    void clear();
    void removeCurrent();
    QList<QAction*> getCategoryActions(QModelIndex subgIdx, QPointF scenePos);

public slots:
    void onNewNodeCreated(QModelIndex subgIdx, const QString& descName, const QPointF& pt);

private:
    IGraphsModel *m_model;
    QMap<QString, ZenoSubGraphScene*> m_scenes;
    QString m_currFile;
};

#endif