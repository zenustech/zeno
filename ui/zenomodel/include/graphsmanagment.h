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
    void importSubGraphs(const QString& fn, const QMap<QString, QString>& map);
    void clear();
    void removeCurrent();
    void appendLog(QtMsgType type, QString fileName, int ln, const QString &msg);
    void appendErr(const QString& nodeName, const QString& msg);
    QGraphicsScene* gvScene(const QModelIndex& subgIdx) const;
    void addScene(const QModelIndex& subgIdx, QGraphicsScene* scene);
    void removeScene(const QString& subgName);
    TIMELINE_INFO timeInfo() const;
    QString zsgPath() const;
    QString zsgDir() const;
    RECORD_SETTING recordSettings() const;
    void setRecordSettings(const RECORD_SETTING& info);
    LAYOUT_SETTING layoutInfo() const;
    void setUserDataInfo(const USERDATA_SETTING& info);
    USERDATA_SETTING userdataInfo();
    bool getDescriptor(const QString &descName, NODE_DESC &desc);
    bool getSubgDesc(const QString& subgName, NODE_DESC& desc);
    bool updateSubgDesc(const QString &descName, const NODE_DESC &desc);
    void renameSubGraph(const QString& oldName, const QString& newName);
    void removeGraph(const QString& subgName);
    void appendSubGraph(const NODE_DESC& desc);
    NODE_DESCS descriptors();
    NODE_CATES getCates();
    NODE_TYPE nodeType(const QString& name);
    QString filePath() const;

signals:
    void modelInited(IGraphsModel* pNodeModel, IGraphsModel* pSubgraphs);
    void modelDataChanged();
    void fileOpened(QString);
    void fileClosed();
    void fileSaved(QString);
    void dirtyChanged(bool);

private slots:
    void onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);

private:
    GraphsManagment(QObject *parent = nullptr);
    void onParseResult(const zenoio::ZSG_PARSE_RESULT& res, IGraphsModel* pNodeModel, IGraphsModel* pSubgraphs);
    NODE_DESCS getCoreDescs();
    void parseDescStr(const QString& descStr, QString& name, QString& type, QVariant& defl);
    void registerCate(const NODE_DESC& desc);
    void initCoreDescriptors();
    void initSubnetDescriptors(const QList<QString>& subgraphs, const zenoio::ZSG_PARSE_RESULT& res);
    void clearSubgDesc();

    NODE_DESCS m_nodesDesc;
    NODE_DESCS m_subgsDesc;
    NODE_CATES m_nodesCate;

    IGraphsModel* m_pNodeModel;
    IGraphsModel* m_pSharedGraphs;
    QStandardItemModel* m_logModel;     //connection with scene.
    mutable std::mutex m_mtx;
    TIMELINE_INFO m_timerInfo;
    RECORD_SETTING m_recordInfo;
    LAYOUT_SETTING m_layoutInfo;
    USERDATA_SETTING m_userdataInfo;
    QMap<QString, QGraphicsScene*> m_scenes;
    QString m_filePath;
};

#endif