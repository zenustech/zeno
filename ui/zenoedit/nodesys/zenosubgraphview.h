#ifndef __ZENO_GRAPHVIEW_H__
#define __ZENO_GRAPHVIEW_H__

#include <QtWidgets>
#include <zenoui/nodesys/nodesys_common.h>

class ZenoSubGraphScene;
class ZenoNewnodeMenu;
class LayerPathWidget;

class _ZenoSubGraphView : public QGraphicsView
{
    Q_OBJECT
    typedef QGraphicsView _base;

public:
    _ZenoSubGraphView(QWidget* parent = nullptr);
    void initScene(ZenoSubGraphScene* pScene);
    void setPath(const QString& path);

protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent* event);
    void wheelEvent(QWheelEvent *event);
    void resizeEvent(QResizeEvent *event);
    void contextMenuEvent(QContextMenuEvent *event) override;
    void drawBackground(QPainter* painter, const QRectF& rect) override;

public slots:
    void redo();
    void undo();
    void copy();
    void paste();
    void find();
    void onSearchResult(SEARCH_RECORD rec);
    void focusOn(const QString& nodeId, const QPointF& pos);

signals:
    void zoomed(qreal);
    void viewChanged(qreal);

private:
    void gentle_zoom(qreal factor);
    void set_modifiers(Qt::KeyboardModifiers modifiers);
    void resetTransform();
    void drawGrid(QPainter* painter, const QRectF& rect);

    QPointF target_scene_pos, target_viewport_pos, m_startPos;
    QPoint m_mousePos;
    QPoint _last_mouse_pos;
    qreal m_factor;
    QString m_path;
    const double m_factor_step = 0.1;
    Qt::KeyboardModifiers _modifiers;
    bool m_dragMove;

    ZenoSubGraphScene* m_scene;
    ZenoNewnodeMenu* m_menu;
};

class LayerPathWidget : public QWidget
{
	Q_OBJECT
public:
    LayerPathWidget(QWidget* parent = nullptr);
	void setPath(const QString& path);
	QString path() const;

signals:
    void pathUpdated(QString);

private slots:
	void onPathItemClicked();

private:
	QString m_path;
};

class ZenoSubGraphView : public QWidget
{
    Q_OBJECT
    typedef QWidget _base;

public:
	ZenoSubGraphView(QWidget* parent = nullptr);
	void initScene(ZenoSubGraphScene* pScene);
	void resetPath(const QString& path, const QString& subGraphName, const QString& objId);

signals:
	void pathUpdated(QString);

private:
    _ZenoSubGraphView* m_view;
    LayerPathWidget* m_pathWidget;
};


#endif