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
class IGraphsModel;

class ZenoSubGraphScene : public QGraphicsScene
{
	Q_OBJECT
public:
    ZenoSubGraphScene(QObject* parent = nullptr);
    ~ZenoSubGraphScene();
    void initModel(IGraphsModel* pGraphsModel, const QModelIndex& index);
    void copy();
    void paste(QPointF pos);
    QRectF nodesBoundingRect() const;
    QModelIndex subGraphIndex() const;
    QModelIndexList selectNodesIndice() const;
    QModelIndexList selectLinkIndice() const;
    void select(const QString& id);
    void select(const std::vector<QString>& nodes);
    void select(const QStringList& nodes);
    void select(const QModelIndexList &nodes);
    void markError(const QString& nodeid);
    void clearMark();
    QGraphicsItem* getNode(const QString& id);
    void collectNodeSelChanged(const QString& ident, bool bSelected);

    // FIXME temp function for merge
    void selectObjViaNodes();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;

public slots:
    void onZoomed(qreal factor);

    void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, int role);
    void onRowsAboutToBeRemoved(const QModelIndex& subgIdx, const QModelIndex &parent, int first, int last);
    void onRowsInserted(const QModelIndex& subgIdx, const QModelIndex& parent, int first, int last);
    void onViewTransformChanged(qreal factor);

    void onLinkInserted(const QModelIndex&, int first, int last);
    void onLinkAboutToBeRemoved(const QModelIndex&, int first, int last);

private slots:
    void clearLayout(const QModelIndex& subGpIdx);
    void onSocketClicked(ZenoSocketItem* pSocketItem);
    void onNodePosChanged();

private:
    void afterSelectionChanged();
    void onSocketAbsorted(const QPointF& mousePos);
    void viewAddLink(const QModelIndex& linkIdx);
    void viewRemoveLink(const QModelIndex& linkIdx);
    void onTempLinkClosed();
    ZenoNode* createNode(const QModelIndex& idx, const NodeUtilParam& params);
    void initLink(const QModelIndex& linkIdx);
    void updateNodeStatus(bool &bOn, int option);

    NodeUtilParam m_nodeParams;
    QPersistentModelIndex m_subgIdx;      //index to the subgraphmodel or node in "graphsModel"
    std::map<QString, ZenoNode*> m_nodes;
    QStringList m_errNodes;        //the nodes which have been marked "error" at run time.
    QMap<QString, ZenoFullLink*> m_links;
    ZenoTempLink* m_tempLink;

    QVector<QPair<QString, bool>> m_selChanges;

    bool m_bOnceOn;
    bool m_bBypassOn;
    bool m_bViewOn;
};

#endif
