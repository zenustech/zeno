#ifndef __GRAPHS_MANAGMENT_H__
#define __GRAPHS_MANAGMENT_H__

#include <QtWidgets>
#include "igraphsmodel.h"

class GraphsManagment : public QObject
{
    Q_OBJECT
public:
    static GraphsManagment& instance();
    ~GraphsManagment();

    void setGraphsModel(IGraphsModel* pNodeModel, IGraphsModel* pSubgraphsModel);
    IGraphsModel* currentModel();
    IGraphsModel* sharedSubgraphs();
    QStandardItemModel* logModel() const;
    IGraphsModel* openZsgFile(const QString &fn);
    bool saveFile(const QString& filePath, APP_SETTINGS settings);
    IGraphsModel* newFile();
    void importGraph(const QString& fn);
    void clear();
    void removeCurrent();
    void appendLog(QtMsgType type, QString fileName, int ln, const QString &msg);
    void appendErr(const QString& nodeName, const QString& msg);
    QGraphicsScene* gvScene(const QModelIndex& subgIdx) const;
    void addScene(const QModelIndex& subgIdx, QGraphicsScene* scene);
    TIMELINE_INFO timeInfo() const;

signals:
    void modelInited(IGraphsModel* pNodeModel, IGraphsModel* pSubgraphs);
    void modelDataChanged();
    void fileOpened(QString);
    void fileClosed();
    void fileSaved(QString);
    void dirtyChanged(bool);

private slots:
    void onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
    void onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    GraphsManagment(QObject *parent = nullptr);

    IGraphsModel* m_pNodeModel;
    IGraphsModel* m_pSharedGraphs;
    QStandardItemModel* m_logModel;     //connection with scene.
    mutable std::mutex m_mtx;
    QString m_currFile;
    TIMELINE_INFO m_timerInfo;
    QMap<QString, QGraphicsScene*> m_scenes;
};

#endif