#ifndef __GRAPHS_MANAGER_H__
#define __GRAPHS_MANAGER_H__

#include <QtWidgets>
#include <zeno/core/data.h>

class AssetsModel;
class GraphsTreeModel;

class GraphsManager : public QObject
{
    Q_OBJECT
public:
    static GraphsManager& instance();
    ~GraphsManager();

    void createGraphs(const zeno::ZSG_PARSE_RESULT ioresult);
    GraphsTreeModel* currentModel() const;
    QStandardItemModel* logModel() const;
    GraphsTreeModel* openZsgFile(const QString &fn);
    bool saveFile(const QString& filePath, APP_SETTINGS settings);
    GraphsTreeModel* newFile();
    void importGraph(const QString& fn);
    void importSubGraphs(const QString& fn, const QMap<QString, QString>& map);
    void clear();
    void removeCurrent();
    void appendLog(QtMsgType type, QString fileName, int ln, const QString &msg);
    void appendErr(const QString& nodeName, const QString& msg);
    QGraphicsScene* gvScene(const QModelIndex& subgIdx) const;
    void addScene(const QModelIndex& subgIdx, QGraphicsScene* scene);
    zeno::TimelineInfo timeInfo() const;
    QString zsgPath() const;
    QString zsgDir() const;

signals:
    void modelInited(GraphsTreeModel*);     //ASSETS?
    void modelDataChanged();
    void fileOpened(QString);
    void fileClosed();
    void fileSaved(QString);
    void dirtyChanged(bool);

private slots:
    void onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
    void onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    GraphsManager(QObject *parent = nullptr);

    GraphsTreeModel* m_model;
    QStandardItemModel* m_logModel;     //connection with scene.
    AssetsModel* m_assets;

    QString m_filePath;

    mutable std::mutex m_mtx;
    zeno::TimelineInfo m_timerInfo;
    QMap<QString, QGraphicsScene*> m_scenes;
};

#endif