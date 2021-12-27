#ifndef __ZENO_GRAPHVIEW_H__
#define __ZENO_GRAPHVIEW_H__

#include <QtWidgets>

class ZenoSubGraphScene;

class ZenoSubGraphView : public QGraphicsView
{
	Q_OBJECT
public:
    ZenoSubGraphView(QWidget* parent = nullptr);
    void setModel(SubGraphModel* pModel);

protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void contextMenuEvent(QContextMenuEvent *event) override;

private slots:
    void onCustomContextMenu(const QPoint& pos);
    void redo();
    void undo();

signals:
    void zoomed(qreal);
    void viewChanged(qreal);

private:
    void gentle_zoom(qreal factor);
    void set_modifiers(Qt::KeyboardModifiers modifiers);
    void zoomIn();
    void zoomOut();
    void resetTransform();
    qreal _factorStep(qreal factor);

    QPointF target_scene_pos, target_viewport_pos, m_startPos;
    qreal m_factor;
    const double m_factor_step = 0.1;
    Qt::KeyboardModifiers _modifiers;
    bool m_dragMove;

    SubGraphModel* m_model; //temp code
    ZenoSubGraphScene* m_scene;
    QMenu* m_menu;
    QAction *m_ctrlz;
    QAction *m_ctrly;
};

#endif