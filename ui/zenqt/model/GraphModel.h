#ifndef __NODESMODEL_H__
#define __NODESMODEL_H__

#include <QObject>
#include <QAbstractItemModel>
#include <QAbstractListModel>
#include <QString>
#include <QQuickItem>
#include "parammodel.h"
#include "LinkModel.h"
#include <zeno/core/Graph.h>
#include <optional>

class GraphModel;
class GraphsTreeModel;

struct NodeItem : public QObject
{
    Q_OBJECT

public:
    //temp cached data for spNode->...
    QString name;
    QString cls;
    QPointF pos;

    std::string m_cbSetPos;
    std::string m_cbSetView;
    std::string m_cbMarkDirty;

    std::weak_ptr<zeno::INode> m_wpNode;
    ParamsModel* params = nullptr;
    bool bView = false;
    bool bCollasped = false;
    bool bDirty = false;

    //for subgraph, but not include assets:
    std::optional<GraphModel*> optSubgraph;

    NodeItem(QObject* parent);
    ~NodeItem();
    void init(GraphModel* pGraphM, std::shared_ptr<zeno::INode> spNode);
    QString getName() {
        return name;
    }

private:
    void unregister();
};

class GraphModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;

    //Q_PROPERTY(CONTROL_TYPE control READ getControl WRITE setControl NOTIFY control_changed)

    QML_ELEMENT

public:
    GraphModel(std::shared_ptr<zeno::Graph> spGraph, GraphsTreeModel* pTree, QObject* parent = nullptr);
    ~GraphModel();
    Q_INVOKABLE LinkModel* getLinkModel() const { return m_linkModel; }
    Q_INVOKABLE int indexFromId(const QString& name) const;
    Q_INVOKABLE void addLink(const QString& fromNodeStr, const QString& fromParamStr,
        const QString& toNodeStr, const QString& toParamStr);
    Q_INVOKABLE QVariant removeLink(const QString& nodeName, const QString& paramName, bool bInput);
    Q_INVOKABLE QString name() const;

    //TEST API
    Q_INVOKABLE QString owner() const;

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QModelIndexList match(const QModelIndex& start, int role,
        const QVariant& value, int hits = 1,
        Qt::MatchFlags flags =
        Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    QHash<int, QByteArray> roleNames() const override;

    //GraphModel:
    zeno::NodeData createNode(const QString& nodeCls, const QString& cate, const QPointF& pos);
    void appendSubgraphNode(QString name, QString cls, NODE_DESCRIPTOR desc, GraphModel* subgraph, const QPointF& pos);
    bool removeNode(const QString& name);
    void setView(const QModelIndex& idx, bool bOn);
    void setMute(const QModelIndex& idx, bool bOn);
    QString updateNodeName(const QModelIndex& idx, QString newName);
    void addLink(const zeno::EdgeInfo& link);
    QList<SEARCH_RESULT> search(const QString& content, SearchType searchType, SearchOpt searchOpts) const;
    GraphModel* getGraphByPath(const QStringList& objPath);
    QModelIndex indexFromUuidPath(const zeno::ObjPath& uuidPath);
    QStringList currentPath() const;
    QModelIndex indexFromName(const QString& name) const;
    void clear();
    void undo();
    void redo();
    void beginTransaction(const QString& name);
    void endTransaction();

    //test functions:
    void updateParamName(QModelIndex nodeIdx, int row, QString newName);
    void syncToAssetsInstance(const QString& assetsName, zeno::ParamsUpdateInfo info);
    void syncToAssetsInstance(const QString& assetsName);
    void updateAssetInstance(const std::shared_ptr<zeno::Graph> spGraph);
    void removeParam(QModelIndex nodeIdx, int row);
    void removeLink(const QModelIndex& linkIdx);
    void removeLink(const zeno::EdgeInfo& link);
    bool updateLink(const QModelIndex& linkIdx, bool bInput, const QString& oldkey, const QString& newkey);
    void moveUpLinkKey(const QModelIndex& linkIdx, bool bInput, const std::string& keyName);
    ParamsModel* params(QModelIndex nodeIdx);
    GraphModel* subgraph(QModelIndex nodeIdx);
    GraphsTreeModel* treeModel() const;
    void setLocked(bool bLocked);
    bool isLocked() const;

signals:
    void reloaded();
    void clearLayout();
    void nameUpdated(const QModelIndex& nodeIdx, const QString& oldName);
    void nodeRemoved(QString nodeId);
    void lockStatusChanged();

private:
    void registerCoreNotify();
    void unRegisterCoreNotify();
    void _appendNode(std::shared_ptr<zeno::INode> spNode);
    void _initLink();
    void _addLink(const zeno::EdgeInfo link);
    bool _removeLink(const zeno::EdgeInfo& edge);
    void _updateName(const QString& oldName, const QString& newName);
    void _clear();
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;

    QString m_graphName;
    QHash<QString, int> m_uuid2Row;
    QHash<int, QString> m_row2uuid;
    QHash<QString, NodeItem*> m_nodes;  //based on uuid
    QHash<QString, QString> m_name2uuid;

    QSet<QString> m_subgNodes;

    std::weak_ptr<zeno::Graph> m_wpCoreGraph;

    std::string m_cbCreateNode;
    std::string m_cbRemoveNode;
    std::string m_cbRenameNode;
    std::string m_cbAddLink;
    std::string m_cbRemoveLink;
    std::string m_cbClearGraph;

    GraphsTreeModel* m_pTree;
    LinkModel* m_linkModel;

    bool m_bLocked = false;

    friend class NodeItem;
};

#endif