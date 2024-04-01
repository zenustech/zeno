#ifndef __THUMBNAIL_VIEW_H__
#define __THUMBNAIL_VIEW_H__

#include <QtWidgets>

class GraphModel;
class ThumbnailScene;
class ZenoSubGraphScene;

class ThumbnailView : public QGraphicsView
{
    Q_OBJECT
    typedef QGraphicsView _base;
public:
    ThumbnailView(QWidget* parent = nullptr);
    void resetScene(ZenoSubGraphScene* pScene);

signals:
    void navigatorChanged(QRectF, QRectF);

protected:
    void drawBackground(QPainter* painter, const QRectF& rect) override;

private:
    ThumbnailScene* m_scene;
};


#endif