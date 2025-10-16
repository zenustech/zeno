#ifndef __GRAPHS_MANAGER_H__
#define __GRAPHS_MANAGER_H__

#include <QtWidgets>
#include <zeno/core/data.h>
#include "uicommon.h"
#include <zeno/io/iocommon.h>
#include <QStandardItemModel>
#include <QQuickItem>

class AssetsModel;
class GraphsTreeModel;
class ZenoSubGraphScene;
class GraphModel;
class NodeCateModel;
class GraphsManager;
class PluginsModel;
class CustomUIModel;

class GraphsManager : public QObject
{
    Q_OBJECT
    QML_ELEMENT
public:
    GraphsManager(QObject* parent = nullptr);
    GraphsManager(const GraphsManager& rhs) = delete;
    ~GraphsManager();

    Q_INVOKABLE void saveProject(const QString& name);
    Q_INVOKABLE void undo(const QStringList& path);
    Q_INVOKABLE void redo(const QStringList& path);
    Q_INVOKABLE QStringList recentFiles() const;
    Q_INVOKABLE void openProject(const QString& zsgpath);
    Q_INVOKABLE void openCustomUIDialog(CustomUIModel* customUIM);
    Q_INVOKABLE void onAssetsCustomUIDialog(const QString& assetsName);
    Q_INVOKABLE void onNodeSelected(const QStringList& graphs_path, const QModelIndex& idx);
    Q_INVOKABLE void addPlugin();
    Q_INVOKABLE void createAssetDialog();
    Q_INVOKABLE void loadAssetDialog();
    Q_INVOKABLE void copy(const QModelIndexList& nodes);
    Q_INVOKABLE QStringList paste(const QPointF& pos, const QStringList& path_of_graphM);
    Q_INVOKABLE QModelIndex getNodeIndexByUuidPath(const QString& objPath);

    //给出节点路径，得到对应节点的索引，如果路径是一个子图节点的路径，就返回子图节点。
    Q_INVOKABLE QModelIndex getNodeIndexByPath(const QString& path);

    Q_PROPERTY(QString currentPath READ currentGraphPath WRITE setCurrentGraphPath NOTIFY currentPathChanged)
    QString currentGraphPath() const;
    void setCurrentGraphPath(const QString& path);

    void createGraphs(const zenoio::ZSG_PARSE_RESULT ioresult);
    GraphModel* mainModel() const { return m_main; }
    inline GraphsTreeModel* currentModel() const { return m_treemodel; }
    AssetsModel* assetsModel() const;
    PluginsModel* pluginModel() const;
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
    NodeCates getCates() const;
    void setIOVersion(zeno::ZSG_VERSION ver);
    void clearMarkOnGv();
    void initRootObjects();

    QQmlComponent* nodeDelegate();
    QQmlComponent* edgeDelegate();

signals:
    void modelInited();
    void modelDataChanged();
    void fileOpened(QString);
    void fileClosed();
    void fileSaved(QString);
    void dirtyChanged(bool);
    void currentPathChanged(QString);

private slots:
    void onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
    void onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    void registerCoreNotify();

    GraphsTreeModel* m_treemodel;
    GraphModel* m_main;
    QStandardItemModel* m_logModel;     //connection with scene.
    AssetsModel* m_assets;
    PluginsModel* m_plugins;

    QString m_filePath;
    QString m_graphPath;

    QQmlComponent* m_nodeDelegate;
    QQmlComponent* m_edgeDelegate;

    mutable std::mutex m_mtx;
    zeno::TimelineInfo m_timerInfo;
    QVector<ZenoSubGraphScene*> m_scenes;
    zeno::ZSG_VERSION m_version;
    bool m_bIniting;
    bool m_bImporting;
};

class GraphsTotalView : public QQuickItem
{
    Q_OBJECT
public:
    GraphsTotalView();
    GraphsTotalView(const GraphsTotalView& graphsview);
    //! Graph that should be displayed in this graph view.
    Q_PROPERTY(GraphsManager* graphsMgr READ getGraphsMgr WRITE setGraphsMgr FINAL)
    void setGraphsMgr(GraphsManager* graph);
    inline GraphsManager* getGraphsMgr() const noexcept { return _graphMgr; }

    Q_INVOKABLE void newFile();
    Q_INVOKABLE void openFile();

signals:
    void modelInited();
    void modelDataChanged();
    void fileOpened(QString);
    void fileClosed();
    void fileSaved(QString);
    void dirtyChanged(bool);

private:
    GraphsManager* _graphMgr = nullptr;
};

#endif