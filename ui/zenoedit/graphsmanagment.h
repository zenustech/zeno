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
    IGraphsModel* newFile();
    void importGraph(const QString &fn);
    void reloadGraph(const QString& graphName);
    bool saveCurrent();
    void clear();
    void removeCurrent();

signals:
    void modelInited(IGraphsModel*);
    void modelDataChanged();

private slots:
    void onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);

private:
    IGraphsModel *m_model;
    GraphsTreeModel* m_pTreeModel;
    QString m_currFile;
};

#endif