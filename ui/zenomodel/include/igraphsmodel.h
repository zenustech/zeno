#ifndef __IGRAPHMODEL_H__
#define __IGRAPHMODEL_H__

#include <QtWidgets>
#include "common.h"
#include "modeldata.h"
#include "modelrole.h"
#include <zenoio/include/common.h>

class LinkModel;
class ViewParamModel;
class SubGraphModel;

class IGraphsModel : public QAbstractItemModel
{
	Q_OBJECT
public:
	explicit IGraphsModel(QObject* parent = nullptr) : QAbstractItemModel(parent) {}

	/* begin: node index: */
	virtual QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override = 0;
	virtual QModelIndex index(const QString& subGraphName) const = 0;
	virtual QModelIndex index(const QString& id, const QModelIndex& subGpIdx) = 0;
	virtual QModelIndex index(int r, const QModelIndex& subGpIdx) = 0;
	virtual QModelIndex nodeIndex(const QString& ident) = 0;
	/* end: node index: */

	virtual QModelIndex nodeIndex(uint32_t sid, uint32_t nodeid) = 0;
	virtual QModelIndex subgIndex(uint32_t sid) = 0;

	virtual int itemCount(const QModelIndex &subGpIdx) const = 0;

	virtual QModelIndex linkIndex(const QModelIndex& subgIdx, int r) = 0;
	virtual QModelIndex linkIndex(const QModelIndex& subgIdx, const QString& outNode, const QString& outSock, const QString& inNode, const QString& inSock) = 0;

	virtual void addNode(const NODE_DATA& nodeData, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;
    virtual void setNodeData(const QModelIndex& nodeIndex, const QModelIndex& subGpIdx, const QVariant& value, int role) = 0;
	virtual void importNodes(
			const QMap<QString, NODE_DATA>& nodes,
			const QList<EdgeInfo>& links,
			const QPointF& pos,
			const QModelIndex& subGpIdx,
			bool enableTransaction = false) = 0;
	virtual void removeNode(const QString& nodeid, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;

	virtual QModelIndex addLink(const QModelIndex& subgIdx, const QModelIndex& fromSock, const QModelIndex& toSock, bool enableTransaction = false) = 0;
	virtual QModelIndex addLink(const QModelIndex& subgIdx, const EdgeInfo& info, bool enableTransaction = false) = 0;
	virtual void removeLink(const QModelIndex& linkIdx, bool enableTransaction = false) = 0;
	virtual void removeLink(const QModelIndex& subgIdx, const EdgeInfo& linkIdx, bool enableTransaction = false) = 0;
	virtual void removeLegacyLink(const QModelIndex& linkIdx) = 0;
	virtual void removeSubGraph(const QString& name) = 0;
	virtual QModelIndex extractSubGraph(const QModelIndexList& nodes, const QModelIndexList& links, const QModelIndex& fromSubg, const QString& toSubg, bool enableTrans = false) = 0;
    virtual bool IsSubGraphNode(const QModelIndex& nodeIdx) const = 0;

	/*
	 fork subnet node indexed by subnetNodeIdx under subgIdx. 
	 */
	virtual QModelIndex fork(const QModelIndex& subgIdx, const QModelIndex& subnetNodeIdx) = 0;
    virtual QModelIndex forkMaterial(const QModelIndex& subgIdx, const QModelIndex& subnetNodeIdx, const QString& subgName, const QString& mtlid, const QString& mtlid_old) = 0;

	virtual void updateParamInfo(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;
	virtual void updateSocketDefl(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;
	virtual void updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info, const QModelIndex& subgIdx, bool enableTransaction = false) = 0;
	virtual void updateBlackboard(const QString& id, const QVariant& blackboard, const QModelIndex& subgIdx, bool enableTransaction) = 0;

	virtual NODE_DATA itemData(const QModelIndex& index, const QModelIndex& subGpIdx) const = 0;
	virtual void setName(const QString& name, const QModelIndex& subGpIdx) = 0;

	virtual NODE_DESCS descriptors() const = 0;
    virtual bool appendSubnetDescsFromZsg(const QList<NODE_DESC>& descs, bool bImport = false) = 0;
	virtual bool getDescriptor(const QString& descName, NODE_DESC& desc) = 0;
	virtual bool updateSubgDesc(const QString& descName, const NODE_DESC& desc) = 0;
	virtual void clearSubGraph(const QModelIndex& subGpIdx) = 0;
	virtual void clear() = 0;
	virtual void undo() = 0;
	virtual void redo() = 0;
	virtual void switchSubGraph(const QString& graphName) {}
	virtual void newSubgraph(const QString& graphName, SUBGRAPH_TYPE type = SUBGRAPH_TYPE::SUBGRAPH_NOR) = 0;
    virtual bool newMaterialSubgraph(const QModelIndex& subgIdx, const QString& graphName, const QPointF& pos) = 0;
	virtual void initMainGraph() = 0;
	virtual void renameSubGraph(const QString& oldName, const QString& newName) = 0;
	virtual bool isDirty() const = 0;
	virtual NODE_CATES getCates() = 0;
	virtual QModelIndexList searchInSubgraph(const QString& objName, const QModelIndex& idx) = 0;
	virtual QModelIndexList subgraphsIndice() const = 0;
    virtual QModelIndexList subgraphsIndice(SUBGRAPH_TYPE type) const = 0;
	virtual QList<SEARCH_RESULT> search(
					const QString& content,
					int searchType,
					int searchOpts,
					QVector<SubGraphModel *> vec = QVector<SubGraphModel *>()) const = 0;
	virtual void removeGraph(int idx) = 0;
	virtual QString fileName() const = 0;
	virtual QString filePath() const = 0;
	virtual void setFilePath(const QString& fn) = 0;
	virtual QRectF viewRect(const QModelIndex& subgIdx) = 0;
	virtual void markDirty() = 0;
	virtual void markNotDescNode() = 0;
	virtual bool hasNotDescNode() const = 0;
	virtual void clearDirty() = 0;
	virtual void collaspe(const QModelIndex& subgIdx) = 0;
	virtual void expand(const QModelIndex& subgIdx) = 0;
    virtual void setIOProcessing(bool bIOProcessing) = 0;
	virtual bool IsIOProcessing() const = 0;
    virtual void beginTransaction(const QString& name) = 0;
    virtual void endTransaction() = 0;
    virtual void beginApiLevel() = 0;
	virtual void endApiLevel() = 0;
	virtual LinkModel* linkModel(const QModelIndex& subgIdx) const = 0;
	virtual LinkModel* legacyLinks(const QModelIndex& subgIdx) const = 0;
	virtual QModelIndexList findSubgraphNode(const QString& subgName) = 0;
	virtual int ModelSetData(
			const QPersistentModelIndex& idx,
			const QVariant& value,
			int role,
			const QString& comment = "") = 0;
	virtual int undoRedo_updateSubgDesc(const QString& descName, const NODE_DESC& desc) = 0;
	virtual QModelIndex indexFromPath(const QString& path) = 0;
	virtual bool addExecuteCommand(QUndoCommand* pCommand) = 0;
	virtual void setIOVersion(zenoio::ZSG_VERSION ver) = 0;
	virtual zenoio::ZSG_VERSION ioVersion() const = 0;
    virtual void setApiRunningEnable(bool bEnable) = 0;
    virtual bool isApiRunningEnable() const = 0;
    virtual bool setCustomName(const QModelIndex &subgIdx, const QModelIndex& Idx, const QString &value) const = 0;
    virtual void markNodeDataChanged(const QModelIndex& idx, bool recursively = true) = 0;
    virtual void clearNodeDataChanged() = 0;
    virtual QStringList subgraphsName() const = 0;

    /*net label*/
    virtual void addNetLabel(const QModelIndex& subgIdx, const QModelIndex& sock, const QString& name) = 0;
    virtual void removeNetLabel(const QModelIndex& subgIdx, const QModelIndex& trigger) = 0;
    virtual void updateNetLabel(const QModelIndex& subgIdx, const QModelIndex& trigger, const QString& oldName, const QString& newName, bool enableTransaction = false) = 0;

    virtual bool addCommandParam(const QString& path, const CommandParam& val) = 0;
    virtual void removeCommandParam(const QString& path) = 0;
    virtual bool updateCommandParam(const QString& path, const CommandParam& val) = 0;
    virtual FuckQMap<QString, CommandParam> commandParams() const = 0;

    virtual QModelIndex getNetOutput(const QModelIndex& subgIdx, const QString& name) const = 0;
	virtual QList<QModelIndex> getNetInputs(const QModelIndex& subgIdx, const QString& name) const = 0;
    virtual QStringList dumpLabels(const QModelIndex& subgIdx) const = 0;

signals:
	void clearLayout2();
	void modelClear();
	void dirtyChanged();
	void pathChanged(const QString& path);
	void reloaded(const QModelIndex& subGpIdx);
	void clearLayout(const QModelIndex& subGpIdx);
	void apiBatchFinished(/*todo: yield msg*/);
	void _dataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
	void _rowsAboutToBeInserted(const QModelIndex& parent, int first, int last);
	void _rowsInserted(const QModelIndex& subGpIdx, const QModelIndex&, int, int);
	void _rowsAboutToBeRemoved(const QModelIndex& subGpIdx, const QModelIndex&, int, int);
	void _rowsRemoved(const QModelIndex& parent, int first, int last);

	void linkDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
	void linkAboutToBeInserted(const QModelIndex& subGpIdx, const QModelIndex& parent, int first, int last);
	void linkInserted(const QModelIndex& subGpIdx, const QModelIndex&, int first, int last);
	void linkAboutToBeRemoved(const QModelIndex& subGpIdx, const QModelIndex&, int first, int last);
	void linkRemoved(const QModelIndex& subGpIdx, const QModelIndex& parent, int first, int last);

    void updateCommandParamSignal(const QString& path);
};


#endif
