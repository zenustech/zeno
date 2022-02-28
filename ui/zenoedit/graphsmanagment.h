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
    void setCurrentModel(IGraphsModel* model);
    IGraphsModel* currentModel();
    GraphsTreeModel* treeModel();
    IGraphsModel* openZsgFile(const QString &fn);
    void importGraph(const QString &fn);
    ZenoSubGraphScene* scene(const QString& subGraphName);
    void reloadGraph(const QString& graphName);
    bool saveCurrent();
    void clear();
    void removeCurrent();
    void initScenes(IGraphsModel* m_model);

private:
    IGraphsModel *m_model;
    QMap<QString, ZenoSubGraphScene*> m_scenes;     //key may be subgraph name or path if treemodel enable.
    GraphsTreeModel* m_pTreeModel;
    QString m_currFile;
};

#endif