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

    void setCurrentModel(IGraphsModel* model);
    IGraphsModel* currentModel();
    QAbstractItemModel* treeModel();
    QStandardItemModel* logModel() const;
    IGraphsModel* openZsgFile(const QString &fn);
    bool saveFile(const QString& filePath, APP_SETTINGS settings);
    IGraphsModel* newFile();
    void importGraph(const QString& fn);
    void importSubGraphs(const QString& fn, const QMap<QString, QString>& map);
    void clear();
    void removeCurrent();
    void appendLog(QtMsgType type, QString fileName, int ln, const QString &msg);
    void appendErr(const QString& nodeName, const QString& msg);
    QGraphicsScene* gvScene(const QModelIndex& subgIdx) const;
    void addScene(const QModelIndex& subgIdx, QGraphicsScene* scene);
    TIMELINE_INFO timeInfo() const;
    QString zsgPath() const;
    QString zsgDir() const;
    RECORD_SETTING recordSettings() const;
    void setRecordSettings(const RECORD_SETTING& info);
    LAYOUT_SETTING layoutInfo() const;
    void setUserDataInfo(const USERDATA_SETTING& info);
    USERDATA_SETTING userdataInfo();

signals:
    void modelInited(IGraphsModel*);
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

    IGraphsModel* m_model;
    QAbstractItemModel* m_pTreeModel;
    QStandardItemModel* m_logModel;     //connection with scene.
    mutable std::mutex m_mtx;
    TIMELINE_INFO m_timerInfo;
    RECORD_SETTING m_recordInfo;
    LAYOUT_SETTING m_layoutInfo;
    USERDATA_SETTING m_userdataInfo;
    QMap<QString, QGraphicsScene*> m_scenes;
};

#endif