#ifndef __ZENO_SUBGRAPH_SCENE_H__
#define __ZENO_SUBGRAPH_SCENE_H__

#include <QtWidgets>
#include <zenoui/render/ztfutil.h>
#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/model/modeldata.h>


class ZenoNode;
class ZenoFullLink;
class ZenoTempLink;
class NodeGridItem;

class ZenoSubGraphScene : public QGraphicsScene
{
	Q_OBJECT
public:
    ZenoSubGraphScene(QObject* parent = nullptr);
    ~ZenoSubGraphScene();
    void initModel(const QModelIndex& index);
    QPointF getSocketPos(bool bInput, const QString &nodeid, const QString &portName);
    void undo();
    void redo();
    void copy();
    void copy2();
    void paste(QPointF pos);
    QRectF nodesBoundingRect() const;
    QModelIndex subGraphIndex() const;
    QModelIndexList selectNodesIndice() const;
    void select(const QString& id);
    void markError(const QString& nodeid);
    void clearMark();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;

public slots:
    void onZoomed(qreal factor);

    void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, int role);
    void onRowsAboutToBeRemoved(const QModelIndex& subgIdx, const QModelIndex &parent, int first, int last);
    void onRowsInserted(const QModelIndex& subgIdx, const QModelIndex& parent, int first, int last);
    void onViewTransformChanged(qreal factor);

	void onLinkDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
	void onLinkAboutToBeInserted(const QModelIndex& subGpIdx, const QModelIndex& parent, int first, int last);
	void onLinkInserted(const QModelIndex& subGpIdx, const QModelIndex&, int first, int last);
	void onLinkAboutToBeRemoved(const QModelIndex& subGpIdx, const QModelIndex&, int first, int last);
	void onLinkRemoved(const QModelIndex& subGpIdx, const QModelIndex& parent, int first, int last);

private slots:
    void reload(const QModelIndex& subGpIdx);
    void clearLayout(const QModelIndex& subGpIdx);

private:
    void updateLinkPos(ZenoNode *pNode, QPointF newPos);
    ZenoNode* createNode(const QModelIndex& idx, const NodeUtilParam& params);

    QRectF m_viewRect;
    NodeUtilParam m_nodeParams;
    QPersistentModelIndex m_subgIdx;      //index to the subgraphmodel or node in "graphsModel"
    std::map<QString, ZenoNode*> m_nodes;
    QList<ZenoNode*> m_errNodes;        //the nodes which have been marked "error" at run time.
    QMap<QString, ZenoFullLink*> m_links;
    ZenoTempLink* m_tempLink;
};

#endif
