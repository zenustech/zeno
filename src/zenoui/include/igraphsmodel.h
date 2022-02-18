#ifndef __IGRAPHMODEL_H__
#define __IGRAPHMODEL_H__

#include <QtWidgets>

#include "../model/modeldata.h"

class IGraphsModel : public QAbstractItemModel
{
	Q_OBJECT
public:
	explicit IGraphsModel(QObject* parent = nullptr) : QAbstractItemModel(parent) {}
	virtual void beginTransaction(const QString& name) = 0;
	virtual void endTransaction() = 0;
	virtual QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const = 0;
	virtual QModelIndex index(const QString& subGraphName) const = 0;
	virtual QModelIndex index(const QString& id, const QModelIndex& subGpIdx) = 0;
	virtual QModelIndex index(int r, const QModelIndex& subGpIdx) = 0;
	virtual QModelIndex linkIndex(int r) = 0;
	virtual QVariant data2(const QModelIndex& subGpIdx, const QModelIndex& index, int role) = 0;
	virtual void setData2(const QModelIndex& subGpIdx, const QModelIndex& index, const QVariant& value, int role) = 0;
	virtual int itemCount(const QModelIndex& subGpIdx) const = 0;
	virtual void addNode(const NODE_DATA& nodeData, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;
	virtual void insertRow(int row, const NODE_DATA& nodeData, const QModelIndex& subGpIdx) = 0;
	virtual void appendNodes(const QList<NODE_DATA>& nodes, const QModelIndex& subGpIdx) = 0;
	virtual void removeNode(const QString& nodeid, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;
	virtual void removeNode(int row, const QModelIndex& subGpIdx) = 0;
	virtual void removeLinks(const QList<QPersistentModelIndex>& info, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;
	virtual void removeLink(const QPersistentModelIndex& linkIdx, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;
	virtual void removeSubGraph(const QString& name) = 0;
	virtual QModelIndex addLink(const EdgeInfo& info, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;
	virtual QVariant getParamValue(const QString& id, const QString& name, const QModelIndex& subGpIdx) = 0;
	virtual void updateParamInfo(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction = false) = 0;
	virtual void updateSocketDefl(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx) = 0;
	virtual void updateSocket(const QString& id, SOCKET_UPDATE_INFO info, const QModelIndex& subGpIdx) = 0;
	virtual void updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info, const QModelIndex& subgIdx, bool enableTransaction = false) = 0;
	virtual NODE_DATA itemData(const QModelIndex& index, const QModelIndex& subGpIdx) const = 0;
	virtual QString name(const QModelIndex& subGpIdx) const = 0;
	virtual void setName(const QString& name, const QModelIndex& subGpIdx) = 0;
	virtual void replaceSubGraphNode(const QString& oldName, const QString& newName, const QModelIndex& subGpIdx) = 0;
	virtual NODES_DATA nodes(const QModelIndex& subGpIdx) = 0;
	virtual NODE_DESCS descriptors() const = 0;
	virtual void setDescriptors(const NODE_DESCS& nodesParams) = 0;
	virtual void clear(const QModelIndex& subGpIdx) = 0;
	virtual void reload(const QModelIndex& subGpIdx) = 0;
	virtual void onModelInited() {};
	virtual void undo() = 0;
	virtual void redo() = 0;

	//GraphsModel legacy?
	virtual void initDescriptors() = 0;
	virtual void switchSubGraph(const QString& graphName) {}
	virtual void newSubgraph(const QString& graphName) = 0;
	virtual void reloadSubGraph(const QString& graphName) = 0;
	virtual void renameSubGraph(const QString& oldName, const QString& newName) = 0;
	virtual bool isDirty() const = 0;
	virtual NODE_CATES getCates() = 0;
	virtual QModelIndexList searchInSubgraph(const QString& objName, const QModelIndex& idx) = 0;
	virtual void removeGraph(int idx) = 0;
	virtual QString fileName() const = 0;

signals:
	void clearLayout2();
	void reloaded(const QModelIndex& subGpIdx);
	void clearLayout(const QModelIndex& subGpIdx);
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
};


#endif