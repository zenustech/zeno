#ifndef PLOTLAYOUT_H
#define PLOTLAYOUT_H

#include <QtWidgets>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_marker.h>

class CurveModel;
struct CURVE_RANGE;
enum SelectType
{
    NoSelected = 0,
    Node_Selected,
    Left_Handle_Selected,
    Right_Handle_Selected
};

class ZPlotCurve : public QwtPlotCurve
{
public:
    explicit ZPlotCurve(const CurveModel* model, const QString& title = QString());
    ~ZPlotCurve();
protected:
    void drawLines(QPainter*,
        const QwtScaleMap& xMap, const QwtScaleMap& yMap,
        const QRectF& canvasRect, int from, int to) const override;
private:
    const CurveModel* m_model;
};

class PlotLayout : public QHBoxLayout
{
    Q_OBJECT
public:
    explicit PlotLayout(QWidget* = nullptr);
    ~PlotLayout();
    void addCurve(CurveModel* model);
    void deleteCurve(const QString& id);
    void updateRange(const CURVE_RANGE& rg);
    const QModelIndex currentIndex() const;
    const CurveModel* currentModel() const;
    void setVisible(const QString& id, bool bVisible);
    QMap<QString, CurveModel*> curveModels();
signals:
    void currentModelChanged(const QString& id);
    void currentIndexChanged(const QModelIndex& index);

protected:
    bool eventFilter(QObject *obj, QEvent *evt) override;

private slots:
    void slotPointSelected(const QPointF& mousePos);
    void slotPointDragged(const QPointF& mousePos);
    void slotSelectionChanged();
    void slotCurrentModelChanged(const QString& id);
    void slotDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);

private:
    int closestPoint(const QPointF& point, double* dist);
    void updateCurve(const CurveModel* pModel);
    void updateHandler();
    void addPlotCurve(const CurveModel* pModel);
private:
    QwtPlot *m_plot;
    QMap<QString, QwtPlotCurve*> m_plotCurves;
    QwtPlotCurve* m_handlerCurve;
    QwtPlotMarker* m_markCurve;
    QModelIndex m_currentIndex;
    SelectType m_selectedType;
    QMap<QString, CurveModel*> m_models;
    CurveModel* m_currentModel;
};


#endif // PLOTLAYOUT_H
