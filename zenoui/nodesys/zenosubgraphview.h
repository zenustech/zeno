#ifndef __ZENO_GRAPHVIEW_H__
#define __ZENO_GRAPHVIEW_H__

#include <QtWidgets>
#include "nodesys_common.h"

class ZenoSubGraphScene;

class ZenoSubGraphView : public QGraphicsView
{
    Q_OBJECT
    typedef QGraphicsView _base;

public:
    ZenoSubGraphView(QWidget* parent = nullptr);
    void setModel(SubGraphModel* pModel);

protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void resizeEvent(QResizeEvent *event);
    void contextMenuEvent(QContextMenuEvent *event) override;

public slots:
    void onCustomContextMenu(const QPoint& pos);
    void redo();
    void undo();
    void copy();
    void paste();
    void find();
    void onSearchResult(SEARCH_RECORD rec);

signals:
    void zoomed(qreal);
    void viewChanged(qreal);

private:
    void gentle_zoom(qreal factor);
    void set_modifiers(Qt::KeyboardModifiers modifiers);
    void zoomIn();
    void zoomOut();
    void resetTransform();
    void _updateSceneRect();
    void _scale(qreal sx, qreal sy, QPointF pos);
    qreal _factorStep(qreal factor);

    QPointF target_scene_pos, target_viewport_pos, m_startPos;
    QPoint m_mousePos;
    QPoint _last_mouse_pos;
    qreal m_factor;
    const double m_factor_step = 0.1;
    Qt::KeyboardModifiers _modifiers;
    bool m_dragMove;

    ZenoSubGraphScene* m_scene;
    QMenu* m_menu;
};

#endif