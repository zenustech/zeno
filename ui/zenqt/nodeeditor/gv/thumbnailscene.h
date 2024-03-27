#ifndef __THUMBNAIL_SCENE_H__
#define __THUMBNAIL_SCENE_H__

#include <QtWidgets>
#include "uicommon.h"

class GraphModel;
class ZenoSubGraphScene;
class ThumbnailNode;
class ZenoNode;

class ThumbnailScene : public QGraphicsScene
{
    Q_OBJECT
    typedef QGraphicsScene _base;

public:
    ThumbnailScene(QRectF sceneRect, QObject* parent = nullptr);
    ~ThumbnailScene();
    void initScene(ZenoSubGraphScene* pScene);
    ZenoSubGraphScene* originalScene() const;

private slots:
    void onNodePosChanged(const ZenoNode* pNode);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;

signals:

private slots:
    void onSceneRectChanged(const QRectF& rc);
    void onNodeInserted(const ZenoNode* pNode);
    void onNodeAboutToRemoved(const ZenoNode* pNode);

private:
    QGraphicsRectItem* onNewThumbNode(const ZenoNode* pNode, const zeno::ObjPath& path);

    QMap<zeno::ObjPath, QGraphicsRectItem*> m_nodes;
    ZenoSubGraphScene* m_origin;
    QTransform m_trans;
    qreal m_scaleX;
    qreal m_scaleY;
};


#endif