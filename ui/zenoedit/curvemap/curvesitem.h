#ifndef __CURVES_ITEM_H__
#define __CURVES_ITEM_H__

class CurveMapView;
class CurveNodeItem;
class CurvePathItem;
class CurveGrid;

/* just a wraper for a full path item with nodes and pathitems.*/
class CurvesItem : public QGraphicsObject
{
    Q_OBJECT
public:
    CurvesItem(CurveMapView* pView, CurveGrid* grid, const QRectF& rc, QGraphicsItem* parent = nullptr);
    ~CurvesItem();
    void initCurves(const QVector<QPointF>& pts, const QVector<QPointF>& handlers);
    int nodeCount() const;
    int indexOf(CurveNodeItem *pItem) const;
    QPointF nodePos(int i) const;
    CurveNodeItem* nodeItem(int i) const;
    QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

signals:
    void nodesDataChanged();

public slots:
    void onNodeGeometryChanged();
	void onNodeDeleted();
    void onPathClicked(const QPointF& pos);

private:
    QVector<CurveNodeItem*> m_vecNodes;
    QVector<CurvePathItem*> m_vecCurves;
    CurveMapView* m_view;
    CurveGrid* m_grid;
};

#endif