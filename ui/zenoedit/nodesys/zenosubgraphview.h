#ifndef __ZENO_GRAPHVIEW_H__
#define __ZENO_GRAPHVIEW_H__

#include <QtWidgets>
#include <zenoui/nodesys/nodesys_common.h>
#include "dock/docktabcontent.h"

class ZenoSubGraphScene;
class ZenoNewnodeMenu;
class LayerPathWidget;
class ZenoSearchBar;

class _ZenoSubGraphView : public QGraphicsView
{
    Q_OBJECT
    typedef QGraphicsView _base;

public:
    _ZenoSubGraphView(QWidget* parent = nullptr);
    void initScene(ZenoSubGraphScene* pScene);
    void setPath(const QString& path);
    qreal scaleFactor() const;
    void setScale(qreal scale);
    void gentle_zoom(qreal factor);
    void showGrid(bool bShow);

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void focusOutEvent(QFocusEvent* event) override;
    void contextMenuEvent(QContextMenuEvent *event) override;
    void drawBackground(QPainter* painter, const QRectF& rect) override;
    bool eventFilter(QObject* watched, QEvent* event) override;
    void scrollContentsBy(int dx, int dy) override;
    void showEvent(QShowEvent *event) override;

public slots:
    void redo();
    void undo();
    void copy();
    void paste();
    void find();
    void esc();
    void cameraFocus();
    void onSearchResult(SEARCH_RECORD rec);
    void focusOn(const QString& nodeId, const QPointF& pos, bool isError);
    void focusOnWithNoSelect(const QString& nodeId);

signals:
    void zoomed(qreal);
    void viewChanged(qreal);

private:
    void set_modifiers(Qt::KeyboardModifiers modifiers);
    void resetTransform();
    void drawGrid(QPainter* painter, const QRectF& rect);
    void scaleBy(qreal scaleFactor);

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
    ZenoSearchBar* m_pSearcher;
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
    ZenoSubGraphScene* scene();
	void resetPath(const QString& path, const QString& subGraphName, const QString& objId, bool isError = false);
    void setZoom(const qreal& scale);
    void focusOnWithNoSelect(const QString& nodeId);
    void focusOn(const QString& nodeId);
    void showFloatPanel(const QModelIndex &subgIdx, const QModelIndexList &nodes);

protected:
    void keyPressEvent(QKeyEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

signals:
	void pathUpdated(QString);
    void zoomed(qreal);

private:
    _ZenoSubGraphView* m_view;
    LayerPathWidget* m_pathWidget;

    QModelIndex m_lastSelectedNode;
    bool m_floatPanelShow;
    DockContent_Parameter *m_prop;
};


#endif
