#ifndef __GRAPHS_MANAGMENT_H__
#define __GRAPHS_MANAGMENT_H__

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
    QAbstractItemModel* treeModel();
    QStandardItemModel* logModel() const;
    IGraphsModel* openZsgFile(const QString &fn);
    IGraphsModel* newFile();
    void importGraph(const QString &fn);
    void reloadGraph(const QString& graphName);
    bool saveCurrent();
    void clear();
    void removeCurrent();
    void appendLog(QtMsgType type, QString fileName, int ln, const QString &msg);
    void appendErr(const QString& nodeName, const QString& msg);
    void appendMsgStream(const QByteArray& arr);
    ZenoSubGraphScene* gvScene(const QModelIndex& subgIdx);

signals:
    void modelInited(IGraphsModel*);
    void modelDataChanged();

private slots:
    void onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
    void onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    IGraphsModel *m_model;
    QAbstractItemModel* m_pTreeModel;
    QStandardItemModel* m_logModel;     //connection with scene.
    mutable QMutex m_mutex;
    QString m_currFile;
    QMap<QString, ZenoSubGraphScene*> m_scenes;
};

#endif