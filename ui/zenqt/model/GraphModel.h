#ifndef __NODESMODEL_H__
#define __NODESMODEL_H__

#include <QObject>
#include <QAbstractItemModel>
#include <QAbstractListModel>
#include <QString>
#include <QQuickItem>
#include <optional>
#include <QUndoCommand>
#include "uicommon.h"


class GraphMImpl;
class GraphModel;
class GraphsTreeModel;
class ParamsModel;
class LinkModel;
struct NodeItem;

class GraphModel : public QAbstractListModel
{
    Q_OBJECT
    typedef QAbstractListModel _base;

    //Q_PROPERTY(CONTROL_TYPE control READ getControl WRITE setControl NOTIFY control_changed)
    QML_ELEMENT

public:
    GraphModel(std::string const& asset_or_maingraph, bool bAsset, GraphsTreeModel* pTree, GraphModel* parentGraph, QObject* parent = nullptr);
    ~GraphModel();
    Q_INVOKABLE LinkModel* getLinkModel() const { return m_linkModel; }
    Q_INVOKABLE int indexFromId(const QString& name) const;
    Q_INVOKABLE QmlParamType::Value getParamType(const QString& node, bool bInput, const QString& param) const;
    Q_INVOKABLE void addLink(
        const QString& fromNodeStr,
        const QString& fromParamStr,
        const QString& toNodeStr,
        const QString& toParamStr,
        bool asListElement = false);

    Q_INVOKABLE bool removeLink(
        const QString& outNode,
        const QString& outParam,
        const QString& inNode,
        const QString& inParam,
        const QString& outKey = "",
        const QString& inKey = "");
    Q_INVOKABLE QString name() const;
    Q_INVOKABLE QStringList path() const;
    Q_INVOKABLE void insertNode(const QString& nodeCls, const QString& cate, const QPointF& pos);
    Q_INVOKABLE bool removeNode(const QString& name);
    Q_INVOKABLE void resetAssetAndLock(const QModelIndex& assetNode);
    Q_INVOKABLE void syncAssetInst(const QModelIndex& assetNode);

    //TEST API
    Q_INVOKABLE QString owner() const;

    Q_PROPERTY(bool lock READ isLocked NOTIFY lockStatusChanged)
    bool isLocked() const;

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
    void setView(const QModelIndex& idx, bool bOn);
    void setBypass(const QModelIndex& idx, bool bOn);
    void setNocache(const QModelIndex& idx, bool bOn);
    void setClearSubnet(const QModelIndex& idx, bool bOn);
    QString updateNodeName(const QModelIndex& idx, QString newName);
    void updateSocketValue(const QModelIndex& nodeidx, const QString socketName, const QVariant newValue);
    void addLink(const zeno::EdgeInfo& link);
    QList<SEARCH_RESULT> search(const QString& content, SearchType searchType, SearchOpt searchOpts);
    QList<SEARCH_RESULT> searchByUuidPath(const zeno::ObjPath& uuidPath);
    QStringList uuidPath2ObjPath(const zeno::ObjPath& uuidPath);
    GraphModel* getGraphByPath(const QStringList& objPath);
    QStringList currentPath() const;
    QModelIndex indexFromName(const QString& name) const;
    QModelIndex indexFromUuid(const QString& uuid) const;
    QModelIndex indexFromUuidPath(const zeno::ObjPath& uuidPath) const;
    void indiceFromUuidPath(const zeno::ObjPath& uuidPath, QModelIndexList& pathNodes) const;
    std::set<std::string> getViewNodePath() const;
    void clear();
    void undo();
    void redo();
    void pushToplevelStack(QUndoCommand* cmd);
    void beginMacro(const QString& name);
    void endMacro();

    //test functions:
    void updateParamName(QModelIndex nodeIdx, int row, QString newName);
    void syncToAssetsInstance_customui(const QString& assetsName, zeno::ParamsUpdateInfo info, const zeno::CustomUI& customui);
    void syncToAssetsInstance(const QString& assetsName);
    void removeParam(QModelIndex nodeIdx, int row);
    void removeLink(const QModelIndex& linkIdx);
    void removeLink(const zeno::EdgeInfo& link);
    bool updateLink(const QModelIndex& linkIdx, bool bInput, const QString& oldkey, const QString& newkey);
    void moveUpLinkKey(const QModelIndex& linkIdx, bool bInput, const std::string& keyName);
    ParamsModel* params(QModelIndex nodeIdx);
    GraphModel* subgraph(QModelIndex nodeIdx);
    GraphsTreeModel* treeModel() const;

    QStringList pasteNodes(const zeno::NodesData& nodes, const zeno::LinksData& links, const QPointF& pos);
    QString name2uuid(const QString& name);

    GraphModel* getTopLevelGraph(const QStringList& currentPath);
    //undo, redo
    zeno::NodeData _createNodeImpl(const QString& cate, zeno::NodeData& nodedata, bool endTransaction = false);
    bool _removeNodeImpl(const QString& name, bool endTransaction = false);
    void _addLink_apicall(const zeno::EdgeInfo& link, bool endTransaction = false);
    void _removeLinkImpl(const zeno::EdgeInfo& link, bool endTransaction = false);
    bool setModelData(const QModelIndex& index, const QVariant& newValue, int role);
    void _setViewImpl(const QModelIndex& idx, bool bOn, bool endTransaction = false);
    void _setByPassImpl(const QModelIndex& idx, bool bOn, bool endTransaction = false);
    void _setNoCacheImpl(const QModelIndex& idx, bool bOn, bool endTransaction = false);
    void _setClearSubnetImpl(const QModelIndex& idx, bool bOn, bool endTransaction = false);

    //unrevertable:
    void clearNodeObjs(const QModelIndex& nodeIdx);
    void clearSubnetObjs(const QModelIndex& nodeIdx);

signals:
    void reloaded();
    void clearLayout();
    void nameUpdated(const QModelIndex& nodeIdx, const QString& oldName);
    void nodeRemoved(QString nodeId);
    void lockStatusChanged();
    void nodesAboutToBeGroup(const QStringList& uuids);

private:
    std::optional<QUndoStack*> m_undoRedoStack;

    void registerCoreNotify();
    void unRegisterCoreNotify();
    void _appendNode(void* spNode);
    void _initLink();
    void _addLink_callback(const zeno::EdgeInfo link);
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

    std::string m_cbCreateNode;
    std::string m_cbRemoveNode;
    std::string m_cbRenameNode;
    std::string m_cbAddLink;
    std::string m_cbRemoveLink;
    std::string m_cbClearGraph;

    GraphsTreeModel* m_pTree;
    LinkModel* m_linkModel;

    QScopedPointer<GraphMImpl> m_impl;
    friend class NodeItem;
};

#endif