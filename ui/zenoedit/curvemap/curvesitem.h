#ifndef __CURVES_ITEM_H__
#define __CURVES_ITEM_H__

class CurveMapView;
class CurveNodeItem;
class CurvePathItem;
class CurveGrid;
class CurveModel;

/* just a wraper for a full path item with nodes and pathitems.*/
class CurvesItem : public QGraphicsObject
{
    Q_OBJECT
public:
    CurvesItem(CurveMapView* pView, CurveGrid* grid, const QRectF& rc, QGraphicsItem* parent = nullptr);
    ~CurvesItem();
    void initCurves(CurveModel* model);
    int nodeCount() const;
    int indexOf(CurveNodeItem *pItem) const;
    void setColor(const QColor& color);
    void _setVisible(bool bVisible);
    QPointF nodePos(int i) const;
    CurveNodeItem* nodeItem(int i) const;
    CurveModel* model() const;
    QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

public slots:
    void onNodeGeometryChanged();
	void onNodeDeleted();
    void onPathClicked(const QPointF& pos);
    void onDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles = QVector<int>());
    void onNodesInserted(const QModelIndex &parent, int first, int last);
    void onNodesAboutToBeRemoved(const QModelIndex &parent, int first, int last);

private:
    QVector<CurveNodeItem*> m_vecNodes;
    QVector<CurvePathItem*> m_vecCurves;
    CurveMapView* m_view;
    CurveGrid* m_grid;
    CurveModel* m_model;
    QColor m_color;
};

#endif