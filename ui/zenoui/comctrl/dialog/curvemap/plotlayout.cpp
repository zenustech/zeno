#include "plotlayout.h"
#include <qwt/qwt_symbol.h>
#include <qwt/qwt_plot_grid.h>
#include <qwt/qwt_scale_div.h>
#include <qwt/qwt_picker_machine.h>
#include <qwt/qwt_legend.h>
#include <qwt/qwt_plot_magnifier.h>
#include <qwt/qwt_plot_panner.h>
#include <qwt/qwt_painter.h>
#include <qwt/qwt_scale_map.h>
#include <qwt/qwt_point_mapper.h>
#include <qwt/qwt_series_data.h>
#include <qwt/qwt_plot_picker.h>
#include <qwt/qwt_scale_engine.h>
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/curvemodel.h>
#include <zenomodel/include/modeldata.h>

static inline double calDist(const QPointF& p1, const QPointF& p2)
{
    double difx = p1.x() - p2.x();
    double dify = p1.y() - p2.y();
    return std::sqrt(difx * difx + dify * dify);
}

static inline double getStepSize(const QwtPlot* plot, QwtAxisId axisId)
{
    auto count = plot->axisMaxMinor(axisId);
    auto ticks = plot->axisScaleDiv(axisId).ticks(QwtScaleDiv::MajorTick);
    double stepSize = 0.01;
    if (ticks.size() > 1)
        stepSize = (ticks[1] - ticks[0]) / (count - 1);
    return stepSize;
}

ZPlotCurve::ZPlotCurve(const CurveModel* model, const QString& title)
    : QwtPlotCurve(title)
    , m_model(model)
{
}

ZPlotCurve::~ZPlotCurve()
{
    m_model = nullptr;
}

void ZPlotCurve::drawLines(QPainter* painter, const QwtScaleMap& xMap, const QwtScaleMap& yMap, const QRectF& canvasRect, int from, int to) const
{
    QwtPointMapper mapper;

    mapper.setFlag(QwtPointMapper::RoundPoints, true);
    mapper.setFlag(QwtPointMapper::WeedOutIntermediatePoints,
        testPaintAttribute(FilterPointsAggressive));

    mapper.setFlag(QwtPointMapper::WeedOutPoints,
        testPaintAttribute(FilterPoints) ||
        testPaintAttribute(FilterPointsAggressive));

    mapper.setBoundingRect(canvasRect);

    const auto& curve = m_model->getItems();
    double distance_x = curve.rg.xTo - curve.rg.xFrom;
    double step = getStepSize(this->plot(), QwtAxis::XBottom) / 10;
    int count = distance_x / step;
    QPolygonF points;
    double x = curve.points.at(0).point.x();
    for (int i = 0; i < count; i++)
    {
        double y = curve.eval(x);
        points << QPointF(x, y);
        x += step;
    }
    QwtPointSeriesData series(points);
    QPolygonF polyline = mapper.toPolygonF(xMap, yMap, &series, from, points.size() - 1);
    QwtPainter::drawPolyline(painter, polyline);
}

PlotLayout::PlotLayout( QWidget *parent)
    : QHBoxLayout(parent)
    , m_plot(nullptr)
    , m_markCurve(nullptr)
    , m_handlerCurve(nullptr)
    , m_currentModel(nullptr)
    , m_selectedType(NoSelected)
{
    m_plot = new QwtPlot;
    m_plot->installEventFilter(this);
    addWidget(m_plot);

    m_plot->setCanvasBackground(QColor(22, 22, 24));
    m_plot->setStyleSheet("color:rgb(204,204,204);");
    QwtPlotMagnifier* magnifier = new QwtPlotMagnifier(m_plot->canvas());
    magnifier->setMouseButton(Qt::MiddleButton);
    QwtPlotPanner* panner = new QwtPlotPanner(m_plot->canvas());
    panner->setMouseButton(Qt::RightButton);

    QwtLegend* legend = new QwtLegend;
    legend->setDefaultItemMode(QwtLegendData::Checkable);
    m_plot->insertLegend(legend, QwtPlot::TopLegend);

    QwtPlotGrid* grid = new QwtPlotGrid;
    grid->setMajorPen(Qt::darkGray, 0, Qt::DotLine);
    grid->attach(m_plot);

    QwtPlotPicker* picker = new QwtPlotPicker(m_plot->canvas());
    picker->setStateMachine(new QwtPickerDragPointMachine());
    picker->setTrackerMode(QwtPicker::AlwaysOff);

    m_handlerCurve = new QwtPlotCurve;
    m_handlerCurve->setStyle(QwtPlotCurve::Lines);
    m_handlerCurve->setVisible(false);
    m_handlerCurve->setItemAttribute(QwtPlotItem::Legend, false);
    m_handlerCurve->attach(m_plot);

    QwtSymbol* symbol = new QwtSymbol(QwtSymbol::Ellipse);
    symbol->setSize(ZenoStyle::dpiScaled(8));
    symbol->setBrush(QBrush(Qt::white, Qt::SolidPattern));
    m_handlerCurve->setPen(QPen(Qt::white, ZenoStyle::dpiScaled(1)));
    m_handlerCurve->setSymbol(symbol);

    symbol = new QwtSymbol(QwtSymbol::Diamond);
    symbol->setSize(ZenoStyle::dpiScaled(12));
    symbol->setBrush(QColor(245, 172, 83));
    m_markCurve = new QwtPlotMarker;
    m_markCurve->setSymbol(symbol);
    m_markCurve->setItemAttribute(QwtPlotItem::Legend, false);
    m_markCurve->attach(m_plot);
    m_markCurve->setVisible(false);

    connect(picker, &QwtPlotPicker::appended, this, &PlotLayout::slotPointSelected);
    connect(picker, &QwtPlotPicker::moved, this, &PlotLayout::slotPointDragged);
    connect(this, &PlotLayout::currentModelChanged, this, &PlotLayout::slotCurrentModelChanged);
}

PlotLayout::~PlotLayout()
{
}

void PlotLayout::addCurve(CurveModel* model)
{
    QString id = model->id();
    m_models.insert(id, model);
    CURVE_RANGE range = model->range();
    m_plot->setAxisScale(QwtAxis::XBottom, range.xFrom, range.xTo);
    m_plot->setAxisScale(QwtAxis::YLeft, range.yFrom, range.yTo);
    addPlotCurve(model);
    connect(model, &CurveModel::dataChanged, this, &PlotLayout::slotDataChanged);
    emit currentModelChanged(model->id());
}

void PlotLayout::slotPointSelected(const QPointF& mousePos)
{
    if (!m_plot->isEnabled())
    {
        return;
    }
    double minDist = getStepSize(m_plot, QwtAxis::XBottom) / 2;
    double dist = minDist / 3;
    int idx = closestPoint(mousePos, &dist);
    if (idx >= 0)
    {
        if (!m_currentModel)
            return;
        if (dist >= minDist)
        {
            double mouse_x = mousePos.x();
            CURVE_DATA curveData = m_currentModel->getItems();
            CURVE_RANGE rg = m_currentModel->range();
            qreal xscale = (rg.xTo - rg.xFrom) / 10.;
            QPointF logicPos(mouse_x, mousePos.y());
            QPointF leftOffset(-xscale, 0);
            QPointF rightOffset(xscale, 0);
            
            m_currentModel->insertRow(idx);
            m_currentIndex = m_currentModel->index(idx, 0);
            m_currentModel->setData(m_currentIndex, HDL_ASYM, ROLE_TYPE);
            m_currentModel->setData(m_currentIndex, logicPos, ROLE_NODEPOS);
            m_currentModel->setData(m_currentIndex, leftOffset, ROLE_LEFTPOS);
            m_currentModel->setData(m_currentIndex, rightOffset, ROLE_RIGHTPOS);
            updateCurve(m_currentModel);
        } else if (dist < minDist / 3)
        {
            m_currentIndex = m_currentModel->index(idx, 0);
        }
        if (m_currentIndex.isValid())
            m_selectedType = Node_Selected;
    }
    if (m_selectedType == Node_Selected)
    {
        slotSelectionChanged();
    }
}

bool PlotLayout::eventFilter(QObject* obj, QEvent* evt)
{
    if (obj == m_plot && evt->type() == QEvent::KeyPress && m_currentIndex.isValid())
    {
        if (QKeyEvent* event = static_cast<QKeyEvent*>(evt))
        {
            if (event->key() == Qt::Key_Delete)
            {
                QAbstractItemModel* pAbstractModel = const_cast<QAbstractItemModel*>(m_currentIndex.model());
                const auto pModel = qobject_cast<CurveModel*>(pAbstractModel);
                if (pModel)
                {
                    pModel->removeRow(m_currentIndex.row());
                    updateCurve(pModel);
                    m_currentIndex = QModelIndex();
                    slotSelectionChanged();
                }
            }
        }
    }
    return false;
}

void PlotLayout::slotPointDragged(const QPointF& mousePos)
{
    if (!m_currentIndex.isValid())
    {
        return;
    }
    QAbstractItemModel* abstractModel = const_cast<QAbstractItemModel*>(m_currentIndex.model());
    const auto& model = dynamic_cast<CurveModel*>(abstractModel);
    bool bLockX = m_currentIndex.data(ROLE_LOCKX).toBool();
    bool bLockY = m_currentIndex.data(ROLE_LOCKY).toBool();
    double mouse_x = mousePos.x();
    double mouse_y = mousePos.y();
    QPointF targetPos;
    //cal y
    double stepSize = getStepSize(m_plot, QwtAxis::YLeft) / 10;
    const auto& rg = model->range();
    if (mouse_y > rg.yFrom)
    {
        if (mouse_y < rg.yTo - stepSize)
        {
            targetPos.setY(mouse_y);
        }
        else
        {
            targetPos.setY(rg.yTo);
        }
    }
    else
    {
        targetPos.setY(rg.yFrom);
    }
    //handle moving
    stepSize = getStepSize(m_plot, QwtAxis::XBottom) / 10;
    if (m_selectedType != Node_Selected && m_handlerCurve->dataSize() == 3)
    {
        const auto& midPos = m_handlerCurve->sample(1);
        const auto& leftPos = m_handlerCurve->sample(0);
        const auto& rightPos = m_handlerCurve->sample(2);
        QPolygonF points;
        if (m_selectedType == Left_Handle_Selected)
        {
            if (calDist(mousePos, leftPos) < stepSize)
            {
                return;
            }
            if (bLockX)
            {
                targetPos.setX(rightPos.x());
            }
            else
            {
                if (midPos.x() - mouse_x > stepSize)
                {
                    targetPos.setX(mouse_x);
                }
                else
                {
                    if (fabs(targetPos.y() - midPos.y()) > stepSize)
                        targetPos.setX(midPos.x());
                    else
                        targetPos.setX(midPos.x() - stepSize);
                }
            }
            if (bLockY)
            {
                targetPos.setY(leftPos.y());
            }

            model->setData(m_currentIndex, targetPos - midPos, ROLE_LEFTPOS);
        }
        else
        {
            if (calDist(mousePos, rightPos) < stepSize)
            {
                return;
            }
            if (bLockX)
            {
                targetPos.setX(rightPos.x());
            }
            else
            {
                if (mouse_x - midPos.x() > stepSize)
                {
                    targetPos.setX(mouse_x);
                }
                else
                {
                    if (fabs(targetPos.y() - midPos.y()) > stepSize)
                        targetPos.setX(midPos.x());
                    else
                        targetPos.setX(midPos.x() + stepSize);
                }
            }
            if (bLockY)
            {
                targetPos.setY(rightPos.y());
            }
            model->setData(m_currentIndex, targetPos - midPos, ROLE_RIGHTPOS);
        }
    }
    else
    {
        //node moving
        const auto& pos = m_currentIndex.data(ROLE_NODEPOS).toPointF();
        targetPos.setX(pos.x());
        if (calDist(mousePos, pos) > stepSize)
        {
            int row = m_currentIndex.row();
            if (row > 0 && row < model->rowCount() - 1)
            {
                auto rightPos = model->index(row + 1, 0).data(ROLE_NODEPOS).toPointF();
                auto leftPos = model->index(row - 1, 0).data(ROLE_NODEPOS).toPointF();
                if (bLockX)
                {
                    targetPos.setX(pos.x());
                }
                else
                {
                    if (mouse_x - leftPos.x() > stepSize)
                    {
                        if (rightPos.x() - mouse_x > stepSize)
                        {
                            targetPos.setX(mouse_x);
                        }
                        else
                        {
                            targetPos.setX(rightPos.x() - stepSize);
                        }
                    }
                    else
                    {
                        targetPos.setX(leftPos.x() + stepSize);
                    }
                }
                if (bLockY)
                {
                    targetPos.setY(pos.y());
                }
            }
        }
        model->setData(m_currentIndex, targetPos, ROLE_NODEPOS);
    }
}

void PlotLayout::updateCurve(const CurveModel* pModel)
{
    QString id = pModel->id();
    if (m_plotCurves.contains(id))
    {
        const auto& curveData = pModel->getItems();
        QPolygonF points;
        for (const auto& p : curveData.points)
        {
            points << p.point;
        }
        m_plotCurves[id]->setSamples(points);
        m_plot->replot();
    }
}


void PlotLayout::slotSelectionChanged()
{
    bool enableEditor = m_currentIndex.isValid();
    m_markCurve->setVisible(enableEditor);
    if (enableEditor)
    {
        QPointF logicPos = m_currentIndex.data(ROLE_NODEPOS).toPointF();
        m_markCurve->setValue(logicPos);
    }
    updateHandler();
    emit currentIndexChanged(m_currentIndex);
}

void PlotLayout::slotCurrentModelChanged(const QString& id)
{
    if (!m_models.contains(id) || (m_currentModel && m_currentModel->id() == id))
        return;
    m_currentModel = m_models[id];
    for (const auto& curve : m_plotCurves)
    {
        QPen pen = curve->pen();
        QColor col = pen.color();
        col.setAlpha(m_plotCurves[id] == curve ? 255 : 50);
        pen.setColor(col);
        curve->setPen(pen);
    }
    m_plot->replot();
    m_currentIndex = QModelIndex();
    slotSelectionChanged();
}

void PlotLayout::slotDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (!topLeft.isValid() || roles.isEmpty())
        return;
    int role = roles.first();
    if (role == ROLE_LOCKY || role == ROLE_LOCKX)
        return;
    QAbstractItemModel* pAbstractModel = const_cast<QAbstractItemModel*>(topLeft.model());
    if (const auto& pModel = qobject_cast<CurveModel*>(pAbstractModel))
    {
        if (role == ROLE_NODEPOS && m_currentIndex == topLeft)
        {
            QPointF logicPos = topLeft.data(ROLE_NODEPOS).toPointF();
            m_markCurve->setValue(logicPos);
        }
        updateCurve(pModel);
        updateHandler();
    }
}

void PlotLayout::updateHandler()
{
    if (m_currentIndex.isValid())
    {
        const auto& leftOffset = m_currentIndex.data(ROLE_LEFTPOS).toPointF();
        const auto& pos = m_currentIndex.data(ROLE_NODEPOS).toPointF();
        const auto& rightOffset = m_currentIndex.data(ROLE_RIGHTPOS).toPointF();
        QPolygonF points;
        if (!leftOffset.isNull())
            points << pos + leftOffset;
        points << pos;
        if (!rightOffset.isNull())
            points << pos + rightOffset;
        m_handlerCurve->setSamples(points);
    }
    m_handlerCurve->setVisible(m_currentIndex.isValid());
    m_plot->replot();
}

void PlotLayout::addPlotCurve(const CurveModel* pModel)
{
    static const QMap<QString, QColor> preset = { {"x", "#CE2F2F"}, {"y", "#2FCD5F"}, {"z", "#307BCD"} };
    QwtSymbol* symbol = new QwtSymbol(QwtSymbol::Diamond);
    symbol->setSize(ZenoStyle::dpiScaled(10));
    symbol->setPen(QPen(Qt::white, ZenoStyle::dpiScaled(1)));
    QString id = pModel->id();
    ZPlotCurve* pPlotCurve = new ZPlotCurve(pModel, id);
    pPlotCurve->attach(m_plot);
    pPlotCurve->setStyle(QwtPlotCurve::Lines);
    pPlotCurve->setSymbol(symbol);
    m_plotCurves[id] = pPlotCurve;
    QColor col;
    if (!preset.contains(id))
        col = QColor("#CE2F2F");
    else
        col = preset[id];
    pPlotCurve->setPen(QPen(col, ZenoStyle::dpiScaled(1)));
    pPlotCurve->setVisible(pModel->getVisible());
    updateCurve(pModel);
}

void PlotLayout::deleteCurve(const QString& id)
{
    if (m_plotCurves.contains(id))
    {
        m_plotCurves[id]->detach();
    }
    m_plot->replot();
    if (m_models.contains(id))
    {
        delete m_models[id];
        m_models.remove(id);
    }
}

void PlotLayout::updateRange(const CURVE_RANGE& newRg)
{
    m_plot->setAxisScale(QwtPlot::xBottom, newRg.xFrom, newRg.xTo);
    m_plot->setAxisScale(QwtPlot::yLeft, newRg.yFrom, newRg.yTo);
    for (auto model : m_models)
    {
        model->resetRange(newRg);
    }
    m_plot->replot();
}

const QModelIndex PlotLayout::currentIndex() const
{
    return m_currentIndex;
}

const CurveModel* PlotLayout::currentModel() const
{
    return m_currentModel;
}

void PlotLayout::setVisible(const QString& id, bool bVisible)
{
    if (m_plotCurves.contains(id))
        m_plotCurves[id]->setVisible(bVisible);
    if (m_models.contains(id))
        m_models[id]->setVisible(bVisible);
    m_plot->replot();
}

QMap<QString, CurveModel*> PlotLayout::curveModels()
{
    return m_models;
}

int PlotLayout::closestPoint(const QPointF& point, double* dist)
{
    if (m_models.isEmpty())
    {
        return -1;
    }

    m_selectedType = NoSelected;
    double min_dist = *dist;
    if (m_handlerCurve->isVisible() && m_handlerCurve->dataSize() == 3)
    {
        QPointF leftPos = m_handlerCurve->sample(0);
        double di = calDist(leftPos, point);
        if (di < min_dist)
        {
            m_selectedType = Left_Handle_Selected;
            return -1;
        }
        else
        {
            QPointF rightPos = m_handlerCurve->sample(2);
            di = calDist(rightPos, point);
            if (di < min_dist)
            {
                m_selectedType = Right_Handle_Selected;
                return -1;
            }
        }
    }
    m_currentIndex = QModelIndex();
    if (!m_currentModel)
    {
        m_selectedType = NoSelected;
        return -1;
    }
    const auto& curve = m_currentModel->getItems();
    int count = m_currentModel->rowCount();
    int idx = count;
    for (int i = 0; i < count; i++)
    {
        const auto& index = m_currentModel->index(i, 0);
        const auto& pos = index.data(ROLE_NODEPOS).value<QPointF>();
        double di = calDist(pos, point);
        if (di < min_dist)
        {
            min_dist = di;
            idx = i;
            break;
        }
        else if (pos.x() > point.x())
        {
            if (i > 0)
            {
                double preDist = point.x() - m_currentModel->index(i - 1, 0).data(ROLE_NODEPOS).value<QPointF>().x();
                double currDist = pos.x() - point.x();
                min_dist = preDist > currDist ? currDist : preDist;
                if (preDist < currDist && preDist < *dist)
                    idx = i - 1;
                else
                    idx = i;
            }
            break;
        }
    }
    if (dist)
        *dist = min_dist;
    return idx;
}
