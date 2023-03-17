#ifndef __ZENO_SUBGRAPH_SCENE_H__
#define __ZENO_SUBGRAPH_SCENE_H__

#include <QtWidgets>
#include <zenoui/render/ztfutil.h>
#include <zenoui/nodesys/nodesys_common.h>
#include <zenomodel/include/modeldata.h>

class ZenoParamWidget;
class ZenoNode;
class ZenoFullLink;
class ZenoTempLink;
class ZenoSocketItem;
class NodeGridItem;

class ZenoSubGraphScene : public QGraphicsScene
{
	Q_OBJECT
public:
    ZenoSubGraphScene(QObject* parent = nullptr);
    ~ZenoSubGraphScene();
    void initModel(const QModelIndex& index);
    void undo();
    void redo();
    void copy();
    void paste(QPointF pos);
    QRectF nodesBoundingRect() const;
    QModelIndex subGraphIndex() const;
    QModelIndexList selectNodesIndice() const;
    QModelIndexList selectLinkIndice() const;
    void select(const QString& id);
    void markError(const QString& nodeid);
    void clearMark();
    QGraphicsItem* getNode(const QString& id);

    // FIXME temp function for merge
    void selectObjViaNodes();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;

public slots:
    void onZoomed(qreal factor);

    void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, int role);
    void onRowsAboutToBeRemoved(const QModelIndex& subgIdx, const QModelIndex &parent, int first, int last);
    void onRowsInserted(const QModelIndex& subgIdx, const QModelIndex& parent, int first, int last);
    void onViewTransformChanged(qreal factor);

    void onLinkInserted(const QModelIndex& subGpIdx, const QModelIndex&, int first, int last);
    void onLinkAboutToBeRemoved(const QModelIndex&, int first, int last);

private slots:
    void reload(const QModelIndex& subGpIdx);
    void clearLayout(const QModelIndex& subGpIdx);
    void onSocketClicked(ZenoSocketItem* pSocketItem);
    void onNodePosChanged();

private:
    void onSocketAbsorted(const QPointF& mousePos);
    void viewAddLink(const QModelIndex& linkIdx);
    void viewRemoveLink(const QModelIndex& linkIdx);
    void onTempLinkClosed();
    ZenoNode* createNode(const QModelIndex& idx, const NodeUtilParam& params);
    void initLink(const QModelIndex& linkIdx);

    NodeUtilParam m_nodeParams;
    QPersistentModelIndex m_subgIdx;      //index to the subgraphmodel or node in "graphsModel"
    std::map<QString, ZenoNode*> m_nodes;
    QStringList m_errNodes;        //the nodes which have been marked "error" at run time.
    QMap<QString, ZenoFullLink*> m_links;
    ZenoTempLink* m_tempLink;
};

#endif
