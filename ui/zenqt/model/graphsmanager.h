#ifndef __GRAPHS_MANAGER_H__
#define __GRAPHS_MANAGER_H__

#include <QtWidgets>
#include <zeno/core/data.h>
#include "uicommon.h"
#include <zeno/io/iocommon.h>
#include <QStandardItemModel>

class AssetsModel;
class GraphsTreeModel;
class ZenoSubGraphScene;
class GraphModel;

class GraphsManager : public QObject
{
    Q_OBJECT
public:
    static GraphsManager& instance();
    ~GraphsManager();

    void createGraphs(const zenoio::ZSG_PARSE_RESULT ioresult);
    inline GraphsTreeModel* currentModel() const { return m_model; }
    AssetsModel* assetsModel() const;
    QStandardItemModel* logModel() const;
    GraphModel* getGraph(const QStringList& objPath) const;
    GraphsTreeModel* openZsgFile(const QString &fn, zenoio::ERR_CODE& code);
    bool saveFile(const QString& filePath, APP_SETTINGS settings);
    bool isInitializing() const;
    bool isImporting() const;
    GraphsTreeModel* newFile();
    void importGraph(const QString& fn);
    void importSubGraphs(const QString& fn, const QMap<QString, QString>& map);
    void clear();
    void removeCurrent();
    void appendLog(QtMsgType type, QString fileName, int ln, const QString &msg);
    void appendErr(const QString& nodeName, const QString& msg);
    void updateAssets(const QString& assetsName, zeno::ParamsUpdateInfo info, const zeno::CustomUI& customui);
    QGraphicsScene* gvScene(const QStringList& graphName) const;
    QGraphicsScene* gvScene(const QModelIndex& subgIdx) const;
    QMap<QStringList, QGraphicsScene*> gvSubScenes(const QStringList& graphName) const;
    void addScene(const QModelIndex& subgIdx, ZenoSubGraphScene* scene);
    void addScene(const QStringList& graphPath, ZenoSubGraphScene* scene);
    bool removeScene(const QStringList& graphPat);
    zeno::TimelineInfo timeInfo() const;
    QString zsgPath() const;
    QString zsgDir() const;
    USERDATA_SETTING userdataInfo() const;
    RECORD_SETTING recordSettings() const;
    zeno::ZSG_VERSION ioVersion() const;
    zeno::NodeCates getCates() const;
    void setIOVersion(zeno::ZSG_VERSION ver);
    void clearMarkOnGv();

signals:
    void modelInited();
    void modelDataChanged();
    void fileOpened(QString);
    void fileClosed();
    void fileSaved(QString);
    void dirtyChanged(bool);
    void assetsAboutToBeRemoved(const QModelIndex&, int, int);

private slots:
    void onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
    void onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    GraphsManager(QObject *parent = nullptr);
    void registerCoreNotify();

    GraphsTreeModel* m_model;
    GraphModel* m_main;
    QStandardItemModel* m_logModel;     //connection with scene.
    AssetsModel* m_assets;

    QString m_filePath;

    mutable std::mutex m_mtx;
    zeno::TimelineInfo m_timerInfo;
    QVector<ZenoSubGraphScene*> m_scenes;
    zeno::ZSG_VERSION m_version;
    bool m_bIniting;
    bool m_bImporting;
};

#endif