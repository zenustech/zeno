#ifndef __GRAPHS_MANAGMENT_H__
#define __GRAPHS_MANAGMENT_H__

class GraphsModel;
class GraphsTreeModel;
class GraphsPlainModel;
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
    GraphsPlainModel* plainModel();
    IGraphsModel* openZsgFile(const QString &fn);
    void importGraph(const QString &fn);
    void reloadGraph(const QString& graphName);
    bool saveCurrent();
    void clear();
    void removeCurrent();

private:
    IGraphsModel *m_model;
    GraphsTreeModel* m_pTreeModel;
    GraphsPlainModel* m_plainModel;
    QString m_currFile;
};

#endif