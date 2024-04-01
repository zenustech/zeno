#ifndef __THUMBNAIL_SCENE_H__
#define __THUMBNAIL_SCENE_H__

#include <QtWidgets>
#include "uicommon.h"

class GraphModel;
class ZenoSubGraphScene;
class ThumbnailNode;
class ZenoNode;


class NavigatorItem : public QGraphicsRectItem
{
    typedef QGraphicsRectItem _base;
public:
    explicit NavigatorItem(QRectF rcView, qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent = nullptr);
    void resize(bool bZoomOut);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

private:
    QRectF m_rcView;
    const qreal cPenWidth = 2;
};


class ThumbnailScene : public QGraphicsScene
{
    Q_OBJECT
    typedef QGraphicsScene _base;

public:
    ThumbnailScene(QRectF sceneRect, QObject* parent = nullptr);
    ~ThumbnailScene();
    void initScene(ZenoSubGraphScene* pScene);
    ZenoSubGraphScene* originalScene() const;
    void onNavigatorPosChanged();

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
    void wheelEvent(QGraphicsSceneWheelEvent* event) override;

signals:
    void navigatorChanged(QRectF, QRectF);

private slots:
    void onSceneRectChanged(const QRectF& rc);
    void onNodeInserted(const ZenoNode* pNode);
    void onNodeAboutToRemoved(const ZenoNode* pNode);

private:
    QGraphicsRectItem* onNewThumbNode(const ZenoNode* pNode, const zeno::ObjPath& path);
    void initNavigator();

    NavigatorItem* m_navigator;
    QMap<zeno::ObjPath, QGraphicsRectItem*> m_nodes;
    ZenoSubGraphScene* m_origin;
    QTransform m_trans;
    qreal m_scaleX;
    qreal m_scaleY;
};


#endif