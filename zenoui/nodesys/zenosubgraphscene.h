#ifndef __ZENO_SUBGRAPH_SCENE_H__
#define __ZENO_SUBGRAPH_SCENE_H__

#include <QtWidgets>
#include "../render/ztfutil.h"

class SubGraphModel;

class ZenoSubGraphScene : public QGraphicsScene
{
	Q_OBJECT
public:
    ZenoSubGraphScene(QObject* parent = nullptr);
    void initModel(SubGraphModel* pModel);

protected:
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event);

public slots:
    void onNewNodeCreated();    //todo: category.
    void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);

private:
    NodeUtilParam m_nodeParams;
	SubGraphModel* m_subgraphModel;
};

#endif